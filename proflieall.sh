FILE_NAME='profilealla100-02.log'

# python profile_cuda_tt_and_cudaevent.py --dataset ogbn-papers100M >> profileall1.log
# python profile_cuda_tt_and_cudaevent.py --dataset ogbn-products >> $FILE_NAME
# python profile_cuda_tt_and_cudaevent.py --dataset ogbn-arxiv>> $FILE_NAME
# python profile_cuda_tt_and_cudaevent.py --dataset yelp >> $FILE_NAME
# python profile_cuda_tt_and_cudaevent.py --dataset reddit >> $FILE_NAME

# python profile_manual_pin_CPUGPU.py --dataset ogbn-papers100M >> profileall1.log
# python profile_manual_pin_CPUGPU.py --dataset ogbn-products >> $FILE_NAME
# python profile_manual_pin_CPUGPU.py --dataset ogbn-arxiv>> $FILE_NAME
# python profile_manual_pin_CPUGPU.py --dataset yelp >> $FILE_NAME
# python profile_manual_pin_CPUGPU.py --dataset reddit >> $FILE_NAME


python profile_cpu_to_gpu_tt_and_cudaevent.py --dataset reddit >> $FILE_NAME
python profile_cpu_to_gpu_tt_and_cudaevent.py --dataset yelp >> $FILE_NAME
python profile_cpu_to_gpu_tt_and_cudaevent.py --dataset ogbn-arxiv >> $FILE_NAME
python profile_cpu_to_gpu_tt_and_cudaevent.py --dataset ogbn-products >> $FILE_NAME
