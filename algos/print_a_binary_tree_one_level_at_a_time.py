from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def print_levels(root):
    if not root:
        return
    
    queue = deque([root])  # Initialize queue with root node
    
    while queue:
        level_size = len(queue)  # Number of nodes at the current level
        level_nodes = []  # Store node values for the current level
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(str(node.val))
            
            # Add child nodes to the queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        print(" ".join(level_nodes))  # Print all nodes at the current level

# Example usage:
# Create a sample tree:
#       1
#      / \
#     2   3
#    / \   \
#   4   5   6

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.right = TreeNode(6)

print_levels(root)

