import numpy as np
from .t_matrix_base import T_Matrix_Base

class AdaptiveMultiplexing(T_Matrix_Base):
    def __init__(self, camera_resolution: tuple[int, int], projector_resolution: tuple[int, int]):
        super().__init__(camera_resolution, projector_resolution)
        self.projector_w = projector_resolution[1]
        self.projector_h = projector_resolution[0]
        
        # state for the rounds
        # start with the root block covering the entire projector
        self.blocks_to_process = [{
            'x': 0, 'y': 0, 'w': self.projector_w, 'h': self.projector_h,
            'id': 0
        }]
        self.next_block_id = 1
        
        # maps camera_pixel_idx -> set of block_ids that illuminate it (used to build the conflict graph)
        self.pixel_to_blocks = {}  # type: dict[int, set[int]]
        
        # stores the final T matrix entries as (camera_pixel_idx, projector_pixel_idx, value)
        self.t_entries = []
        
        self.current_phase_blocks = []
        self.current_phase_colors = {}
        self.finished = False
        
        # brightness threshold, set in the first round
        self.threshold_map = None

    def get_next_patterns(self) -> list[np.ndarray]:
        """
        Returns this round's patterns: blocks that never light the same camera pixel
        share a pattern and are projected together.
        """
        if not self.blocks_to_process:
            self.finished = True
            return []

        blocks = self.blocks_to_process
        n = len(blocks)
        
        # adj is the conflict graph (block idx to set of blocks indices)
        adj = {i: set() for i in range(n)}
        block_id_to_idx = {b['id']: i for i, b in enumerate(blocks)}
        
        # only iterate over pixels lit by current blocks, not all camera pixels
        for pix_idx, block_ids in self.pixel_to_blocks.items():
            # grab current block indexes (indexes to blocks to process) that hit this pixel
            active_indices = [block_id_to_idx[bid] for bid in block_ids if bid in block_id_to_idx]
            
            if len(active_indices) > 1:
                # all pairs in active_indices conflict
                for i in range(len(active_indices)):
                    for j in range(i + 1, len(active_indices)):
                        u, v = active_indices[i], active_indices[j]
                        adj[u].add(v)
                        adj[v].add(u)

        # Greedy graph coloring: each color becomes one pattern
        colors = {} # block_idx -> color_idx
        for i in range(n):
            neighbor_colors = {colors[neighbor] for neighbor in adj[i] if neighbor in colors}
            color = 0
            while color in neighbor_colors:
                color += 1
            colors[i] = color
            
        num_colors = max(colors.values()) + 1 if colors else 0
        
        # generate concurrent patterns if they have the same color
        patterns = []
        self.current_phase_blocks = blocks
        self.current_phase_colors = colors
        
        for c in range(num_colors):
            pattern = np.zeros(self.projector_resolution)
            for i in range(n):
                if colors[i] == c:
                    b = blocks[i]
                    pattern[b['y']:b['y']+b['h'], b['x']:b['x']+b['w']] = 1
            patterns.append(pattern)
            
        return patterns

    def process_captured_images(self, images: list[np.ndarray], threshold=0.01):
        """
        Uses this round's captures to either store T entries (single-pixel blocks)
        or subdivide blocks and update pixel_to_blocks for the next round.
        """
        if not self.current_phase_blocks:
            return

        blocks = self.current_phase_blocks
        colors = self.current_phase_colors
        
        # Flatten images to ensure consistent indexing
        flat_images = [img.reshape(-1) for img in images]
        
        block_children_map = {} # block_id -> list of children
        unique_next_blocks = {} # child_id -> child_struct

        # either subdivide and update pixel_to_blocks, or store T matrix entries
        for i, block in enumerate(blocks):
            img_idx = colors[i]  # each color is one projected pattern
            if img_idx >= len(flat_images): continue
            
            img = flat_images[img_idx]
            
            # per-pixel alternative (disabled): np.maximum(img * 0.0001, threshold)
            if block['id'] == 0:
                self.threshold_map = threshold
            
            # Determine which threshold to use
            current_threshold = self.threshold_map if self.threshold_map is not None else threshold
            
            # Find pixels lit by this pattern
            lit_indices = np.where(img > current_threshold)[0]
            
            if len(lit_indices) == 0:
                continue
                
            # If this is the root block, initialize the pixel map
            if block['id'] == 0:
                for pix_idx in lit_indices:
                    if pix_idx not in self.pixel_to_blocks:
                        self.pixel_to_blocks[pix_idx] = set()
                    self.pixel_to_blocks[pix_idx].add(0)

            # only pixels this block was expected to light (no other block in this pattern lights them)
            relevant_pixels = []
            if block['id'] == 0:
                relevant_pixels = lit_indices
            else:
                relevant_pixels = [p for p in lit_indices if block['id'] in self.pixel_to_blocks.get(p, set())]

            if len(relevant_pixels) == 0:
                continue

            # subdivide or store decision
            is_pixel_level = (block['w'] <= 1 and block['h'] <= 1)
            
            if is_pixel_level:
                # leaf node so store T matrix entry
                p_idx = block['y'] * self.projector_resolution[1] + block['x']
                for pix_idx in relevant_pixels:
                    val = img[pix_idx]
                    self.t_entries.append((pix_idx, p_idx, val))
                    # Remove from pixel map as it is resolved
                    if pix_idx in self.pixel_to_blocks:
                         self.pixel_to_blocks[pix_idx].discard(block['id'])
                         if not self.pixel_to_blocks[pix_idx]:
                             del self.pixel_to_blocks[pix_idx]
            else:
                # subdivide: in pixel_to_blocks, replace this block with its children
                if block['id'] not in block_children_map:
                    block_children_map[block['id']] = self._subdivide(block)
                
                children = block_children_map[block['id']]

                for pix_idx in relevant_pixels:
                    if pix_idx in self.pixel_to_blocks:
                        self.pixel_to_blocks[pix_idx].discard(block['id'])
                        for child in children:
                            self.pixel_to_blocks[pix_idx].add(child['id'])
                            
                # get children blocks for next round
                for child in children:
                    unique_next_blocks[child['id']] = child

        # only keep blocks that are pointed to by at least one pixel
        active_block_ids = set()
        for block_ids in self.pixel_to_blocks.values():
            active_block_ids.update(block_ids)
            
        self.blocks_to_process = [b for b_id, b in unique_next_blocks.items() if b_id in active_block_ids]

    def _subdivide(self, block):
        w, h = block['w'], block['h']
        x, y = block['x'], block['y']
        
        # split roughly in half
        w1 = w // 2
        w2 = w - w1
        h1 = h // 2
        h2 = h - h1
        
        children = []
        # Top-left, Top-right, Bottom-left, Bottom-right
        coords = [
            (x, y, w1, h1), (x + w1, y, w2, h1),
            (x, y + h1, w1, h2), (x + w1, y + h1, w2, h2)
        ]
        
        for (cx, cy, cw, ch) in coords:
            if cw > 0 and ch > 0:
                children.append({
                    'x': cx, 'y': cy, 'w': cw, 'h': ch,
                    'id': self.next_block_id
                })
                self.next_block_id += 1
        return children

    def get_T_matrix(self) -> np.ndarray:
        
        num_c = int(np.prod(self.camera_resolution))
        num_p = self.projector_resolution[0] * self.projector_resolution[1]
        T = np.zeros((num_c, num_p))
        
        for pix_idx, p_idx, val in self.t_entries:
            if 0 <= pix_idx < num_c and 0 <= p_idx < num_p:
                T[pix_idx, p_idx] = val
        return T
