import os
from pathlib import Path 

dataset = 'mov100k'
assert dataset in ['Ciao', 'epinions', 'mov100k', 'Synthetic', 'Synthetic-r-10', 'Synthetic-r-200', 'Synthetic-r-4-NL']

meta_path = str(Path(__file__).resolve().parent)

main_path = meta_path + '/datas/' + dataset + '/'


file_path = [os.path.join(main_path, '%s.txt' % str(i)) for i in range(1, 6)]
file_path = ','.join(file_path)

if dataset in ['Synthetic-r-10', 'Synthetic-r-200', 'Synthetic-r-4-NL', 'Synthetic']:
    file_path = meta_path + '/datas/' + dataset + '.txt'

if dataset == 'gowalla':
    file_path = meta_path + '/datas/' + dataset + '.txt'


model_path = meta_path + '/models/'