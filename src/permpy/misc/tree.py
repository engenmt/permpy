class Node(object):
    """A class for nodes of binary plane trees."""

    def __init__(self, label):
        self.label = label
        self.left_child = None
        self.right_child = None

    def __len__(self):
        length = 1
        if self.left_child:
            length += len(self.left_child)
        if self.right_child:
            length += len(self.right_child)
        return length

    def __repr__(self, width=None):
        label = str(self.label)

        if width:
            label = f"{label:{width}s}"
        else:
            width = len(label)

        lines = [label]

        if self.right_child:
            right_repr = self.right_child.__repr__(width=width)
            right_lines = right_repr.split("\n")
            for idx, right_line in enumerate(right_lines):
                try:
                    lines[idx] += " - " + right_line
                except IndexError:
                    lines.append(" " * (width + 3))

        if self.left_child:
            # seen = len(lines)
            left_repr = self.left_child.__repr__(width=width)
            left_lines = left_repr.split("\n")
            for idx, left_line in enumerate(left_lines):
                if idx == 0:
                    lines.append(" " * (width) + r" \ " + left_line)
                else:
                    lines.append(" " * (width + 3) + left_line)

        return "\n".join(lines)

    def width(self):
        raise NotImplementedError

    def depth(self):
        depth_below = 0
        if self.left_child:
            depth_below = max(depth_below, self.left_child.depth())
        if self.right_child:
            depth_below = max(depth_below, self.right_child.depth())
        return depth_below + 1

    def add_left(self, child):
        self.left_child = child

    def add_right(self, child):
        self.right_child = child

    def has_children(self):
        return self.left_child or self.right_child

    def postorder(self):
        reading = []
        if self.left_child:
            reading += self.left_child.postorder()
        if self.right_child:
            reading += self.right_child.postorder()
        reading.append(self.label)
        return reading


def create_tree(p):
    """Given a nonempty permutation `p`, return the associated decreasing binary plane tree."""
    L = list(p)

    max_val = max(p)
    max_idx = p.index(max_val)

    before = L[:max_idx]
    # print(f"before={before}")
    after = L[max_idx + 1 :]
    # print(f"after={after}")

    T = Node(max_val)
    if len(before):
        T.add_left(create_tree(before))
    if len(after):
        T.add_right(create_tree(after))

    return T


if __name__ == "__main__":
    p = [1, 3, 2, 7, 6, 5, 4]
    print(p)
    t = create_tree(p)
    print(t)
    print(t.postorder())

    p = [2, 3, 1]
    print(p)
    t = create_tree(p)
    print(t)
    p = t.postorder()
    print(p)
    t = create_tree(p)
    print(t.postorder())
