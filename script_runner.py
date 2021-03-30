import runpy
from misc.misc import get_time

t1 = get_time()
print('Scripts start at {0} ... '.format(t1))

runpy.run_path('0_clean_raw.py')

t2 = get_time()
print('Total exe time: {0}'.format(t2-t1))
