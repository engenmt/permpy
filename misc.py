''' This is working code which may or may not be added to a class at some point.'''

from permpy import *

def color_4231(p):
  red_entries = []
  blue_entries = []

  for entry in p:
    if Permutation(blue_entries+[entry]).involves(Permutation(312)):
      red_entries.append(entry)
    else:
      blue_entries.append(entry) 

  return (red_entries, blue_entries)

def label_4231(p, new_version=False):
  (red_entries, blue_entries) = color_4231(p)
  word = ''
  for (index, entry) in enumerate(p):
    if new_version and index in p.ltrmax():
      word += 'D'
    elif new_version and index in p.rtlmin():
      word += 'A'
    elif entry in red_entries:
      word += ('A' if index in p.rtlmin() else 'B')
    else:
      word += ('D' if index in p.ltrmax() else 'C')
  return (word, ''.join([word[i] for i in p.inverse()]))

def color_1324(p):
  red_entries = []
  blue_entries = []

  for entry in p:
    if (len(blue_entries) > 0 and entry > min(blue_entries)) or Permutation(red_entries+[entry]).involves(Permutation(132)):
      blue_entries.append(entry)
    else:
      red_entries.append(entry) 

  return (red_entries, blue_entries)

def label_1324(p, new_version=False):
  (red_entries, blue_entries) = color_1324(p)
  word = ''
  for (index, entry) in enumerate(p):
    if new_version and index in p.rtlmax():
      word += 'D'
    elif entry in red_entries:
      word += ('A' if index in p.ltrmin() else 'B')
    else:
      word += ('D' if index in p.rtlmax() else 'C')
  return (word, ''.join([word[i] for i in p.inverse()]))

def check_pattern(word_pairs, pattern):
  n = [0,0]
  for (x,y) in word_pairs:
    if x.find(pattern) != -1:
      n[0] += 1
    if y.find(pattern) != -1:
      n[1] += 1
  if n[0] == 0 or n[1] == 0:
    print('\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\t\t'+pattern+': ('+str(n[0])+', '+str(n[1])+')\n\n')
  return n

def check_pattern_list(word_pairs, patterns):
  d = dict()
  for pattern in patterns:
    d[pattern] = check_pattern(word_pairs, pattern)
  return d

def allstrings(alphabet, length):
  c = []
  for i in range(length):
    c = [[x]+y for x in alphabet for y in c or [[]]]
  return [''.join(a) for a in c]

def nc_contains(w,u):
  seen = 0
  for i in range(0, len(w)):
    if w[i] == u[seen]:
      seen += 1
      if seen == len(u):
        return True
  return False

def check_nc_pattern(word_pairs, pattern):
  n = [0,0]
  for (x,y) in word_pairs:
    if nc_contains(x,pattern):
      n[0] += 1
    if nc_contains(y,pattern):
      n[1] += 1
  return n

def check_nc_pattern_list(word_pairs, patterns):
  d = dict()
  for pattern in patterns:
    d[pattern] = check_nc_pattern(word_pairs, pattern)
  return d