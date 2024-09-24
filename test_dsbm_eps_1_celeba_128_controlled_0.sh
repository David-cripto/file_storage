# exp_name=celeba_male2female_dsbm_params_minibatch_ot,first_coupling=ind,first_num_iter=100000,gamma_max=0.01,gamma_min=0.01,method=dbdsb,num_iter=20000,num_steps=100,path_to_save_info=

path_to_DSBM="/cache/david/dsbm_code/DSBM_AdvBM"
data_dataset="celeba_male_to_female"
input_dataset="celeba_male"
label_output="celeba_female"
path_to_save_info="/cache/david/dsbm_code/DSBM_AdvBM/experiments"

exp_description="exp_name=celeba_male2female_dsbm_params_minibatch_ot_eps_1,first_coupling=ref,first_num_iter=20000,gamma_max=0.01,gamma_min=0.01,method=dbdsb,num_iter=20000,num_steps=100,path_to_save_info="
# exp_description="exp_name=celeba_male2female_128_dsbm_params_minibatch_ot,first_coupling=ind,first_num_iter=200000,gamma_max=0.01,gamma_min=0.01,method=dbdsb,num_iter=20000,num_steps=100,path_to_save_info=/trinity/home/sergei.kholkin/DSBM_AdvBM/experiments,use_minibatch_ot=True"
# exp_description="exp_name=celeba_male2female_128_dsbm_params_minibatch_ot,first_coupling=ind,first_num_iter=200000,gamma_max=0.01,gamma_min=0.01,method=dbdsb,num_iter=20000,num_steps=100,path_to_save_info="
# exp_description="exp_name=celeba_male2female_dsbm_params_minibatch_ot,first_coupling=ind,first_num_iter=100000,gamma_max=0.01,gamma_min=0.01,method=dbdsb,num_iter=20000,num_steps=100,path_to_save_info="
# path_to_save_info="/cache/selikhanovych/DSBM_AdvBM/saved_info/bsdm"
python my_test_dbdsb.py --path_to_DSBM ${path_to_DSBM} --data_dataset ${data_dataset} \
--input_dataset ${input_dataset} --exp_description ${exp_description} \
--path_to_save_info ${path_to_save_info} --label_output ${label_output} \
--imf_iters 1 --imf_iters 2 --imf_iters 3 --imf_iters 4 --imf_iters 5 --imf_iters 6 --imf_iters 7 --imf_iters 8 --imf_iters 9 --imf_iters 10 --imf_iters 11 --imf_iters 12 --imf_iters 13 --imf_iters 14 --imf_iters 15 --imf_iters 16 --imf_iters 17 --imf_iters 18 --imf_iters 19 --imf_iters 20 \
--fb_eval 'f' --fb_eval 'b' \
--nfe 100
