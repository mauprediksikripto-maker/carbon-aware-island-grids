import sys, json, re
src, dst = sys.argv[1], sys.argv[2]
L = open(src).read().split('\n')
cells, cur, kind = [], [], None
def flush():
    if kind is None: return
    body = cur[:]
    while body and not body[-1].strip(): body.pop()
    while body and not body[0].strip(): body.pop(0)
    if kind == 'md':
        body = [re.sub(r'^# ?', '', l) for l in body]
        cells.append(dict(cell_type='markdown', metadata={}, source=[l + '\n' for l in body[:-1]] + body[-1:]))
    else:
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                          source=[l + '\n' for l in body[:-1]] + body[-1:]))
for l in L:
    if l.startswith('# %%'):
        flush(); cur = []; kind = 'md' if '[md]' in l else 'code'
    else:
        cur.append(l)
flush()
nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
          language_info=dict(name='python'), colab=dict(provenance=[])), nbformat=4, nbformat_minor=5)
json.dump(nb, open(dst, 'w'), indent=1)
print(len(cells), 'cells ->', dst)
