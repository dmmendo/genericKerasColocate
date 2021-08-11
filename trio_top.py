import os
from itertools import combinations

interval_t = 0.002
run_count = 100
query_rates = [0.5,1,2]
dataset_name = 'vggstyle_cnn_v2'
dataset_dir = '/home/danielmendoza865/gen_models/'
arrival_dist_dir = '/home/danielmendoza865/poisson_arrival/'
dataset_ex = [0,18,23,40,47,49,37,21,23,43]


for q_r in query_rates:
    arrival_dist_file = arrival_dist_dir+'qrate-'+str(q_r)+'_interval-'+str(int(interval_t*1000))+'ms_tcount-'+str(run_count)+'_0.poisson'
    for entry in combinations(dataset_ex,3):
        i,j,k = entry
        model_file = dataset_dir+dataset_name+'_'+str(i)+'/1' 

        col_model0_file = dataset_dir+dataset_name+'_'+str(j)+'/1'
        col_model1_file = dataset_dir+dataset_name+'_'+str(k)+'/1'

        out_file = 'vggstyle_cnn_v2_qrate-'+str(q_r)+'_interval-'+str(interval_t*1000)+'_tcount-'+str(run_count)+'_0.runtime_seq'
        f = open(out_file,'a')
        f.write(str(i)+','+str(j)+','+str(k)+' ')
        f.close()

        cmd_str = 'python trio_colocate_run.py '+model_file+' '+col_model0_file+' '+col_model1_file+' '
        cmd_str += arrival_dist_file+' '+out_file+' '+str(interval_t)+' '+str(run_count)
        print(cmd_str)
        os.system(cmd_str)

            

