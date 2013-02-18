from permutation import *

class AvoidenceGrid:
  '''  0 = allowed
       1 = forbidden by basis
       2 = forbidden by user '''

  def __init__(self, basis=[], ag=[], P=Permutation([]), pregenerate_length = 0):
    self.basis = basis
    self.P = Permutation([])
    self.placement_grid = [[0]]

  def __len__(self):
    return len(self.P)

  def add_entry(self, new_index, new_entry):
    L = list(self.P)
    L.insert(new_index, new_entry-.5)
    self.P = Permutation(L)

    old_length = len(self.placement_grid)
    self.placement_grid = self.extend_grid(self.placement_grid, new_index, new_entry)

    self.update_grid()

  def update_grid(self):
    ''' We assume that the grid has been updated each time a new element
              is inserted, so that when this function is called, we only
              need to check the new boxes created by that element's
              insertion.'''
    new_placement_grid = self.placement_grid
    for i in range(0, len(new_placement_grid)):
      for j in range(0, len(new_placement_grid)):
        if new_placement_grid[i][j] == 0:
          ''' need to put in a new element here and see if we violate the basis '''
          L = list(self.P)
          L.insert(i, j-.5)
          new_permutation = Permutation(L)
          for b in self.basis:
            if b.involved_in(new_permutation):
              # print '(',i,', ',j,'): ',new_permutation," not allowed because of ",b
              new_placement_grid[i][j] = 1
              break
    self.placement_grid = new_placement_grid

  def test_for_simple_extension(self, P, max_length):
    self.P = P
    while len(self.placement_grid) <= len(P):
      self.placement_grid = self.extend_grid(self.placement_grid, 0,0)
    self.update_grid()

    c = self.check_for_blocks()
    if c[0]:
      print 'The interval of size',c[2],'starting at index',c[1],'is blocked.'
    return True

    intervals = self.P.all_intervals()

    

  @staticmethod
  def extend_grid(placement_grid, new_index, new_entry):
    new_placement_grid = []
    for i in range(0, len(placement_grid)):
      r = placement_grid[i][:]
      r.insert(new_index, r[new_index])
      new_placement_grid.append(r)
      if i == new_entry:
        new_placement_grid.append(r[:])
    return new_placement_grid

  def __repr__(self):
    s = '';
    Q = self.P.inverse()
    for i in reversed(range(len(self)+1)):
      for j in range(len(self)+1):
        s += ('o ' if self.placement_grid[j][i] == 0 else '- ')
      s += '\n'
      if i > 0:
        w = 2*Q[i-1]+1
        s += ' '*w
        s += '*\n'
    return s

  def check_for_blocks(self):
    length = len(self.P)
    all_blocks = self.P.all_intervals()
    for block_size in range(0,len(all_blocks)):
      for simple_block in all_blocks[block_size]:
        index_range = range(simple_block, simple_block+block_size)
        values = list(self.P[simple_block:simple_block+block_size])

        # print 'Block for indices ',index_range,' and values ',values
        # if block_size == 2:
        #   in_indices = set(range(simple_block, simple_block+block_size+1))
        #   in_values = set(range(min(values), max(values)+2))
        # else:
        in_indices = range(simple_block, simple_block+block_size+1)
        in_values = range(min(values), max(values)+2)
        # out_indices = set(range(0,length+1)).difference(in_indices)
        # out_values = set(range(0,length+1)).difference(in_values)

        block = True
        # print 'simple_block: ',simple_block
        # print 'block_size: ',block_size
        # print 'in_indices: ',in_indices
        # print 'in_values: ',in_values
        for i in range(0,length+1):
          s = ''
          quit_early = False
          for j in range(0, length+1):
            if (j in in_indices) != (i in in_values):
              # print '\t\tLooking at (j,i) = (',j,',',i,')'
              if block_size == 2:
                if j != in_indices[1] and i != in_values[1]:
                  continue
              # print '\t\tChecking (column,row) = (',j,',',i,'), which has value ',self.placement_grid[j][i]
              if self.placement_grid[j][i] == 0:
                quit_early = True
                block = False
                break
          if quit_early:
            break
        if block:
          # print 'The interval of size ',block_size,' starting at index ',simple_block,' is blocked!'
          return (True, simple_block, block_size)
    return (False)









