from ..permutation import *
from ..avclass import *


class RestrictedContainer(object):
    def __init__(self, basis_list, input_list, container=[], output_list=[]):
        self.input_list = (
            input_list if type(input_list) == list else range(1, input_list + 1)
        )
        self.output_list = output_list
        self.basis_list = [Permutation(b) for b in basis_list]
        self.container = container

    def push(self, position):
        if len(self.input_list) == 0:
            return -1
        new_container = (
            self.container[:position] + [self.input_list[0]] + self.container[position:]
        )
        for basis in self.basis_list:
            if basis.involved_in(Permutation(new_container)):
                return -1
        else:
            return RestrictedContainer(
                self.basis_list, self.input_list[1:], new_container, self.output_list
            )

    def pop(self):
        if len(self.container) == 0:
            return -1
        popped = self.container[0]
        new_output = self.output_list + [popped]
        return RestrictedContainer(
            self.basis_list, self.input_list, self.container[1:], new_output
        )

    def __repr__(self):
        return f"{self.output_list} | {self.container} | {self.input_list}"

    def container_size(self):
        return len(self.container)

    def output_size(self):
        return len(self.input_list)

    def input_size(self):
        return len(self.input_list)

    def output(self):
        return self.output_list
