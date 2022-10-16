@REM python train.py --experiment unet_all_sets
@REM python predict.py --experiment unet_all_sets

python train.py --experiment baseline_all_sets
python predict.py --experiment baseline_all_sets

python train.py --experiment vq_all_sets
python predict.py --experiment vq_all_sets

python train.py --experiment dkvb_all_sets
python predict.py --experiment dkvb_all_sets

python train.py --experiment vq_all_sets_64
python predict.py --experiment vq_all_sets_64

python train.py --experiment dkvb_all_sets_64
python predict.py --experiment dkvb_all_sets_64

python train.py --experiment vq_all_sets_128
python predict.py --experiment vq_all_sets_128

python train.py --experiment dkvb_all_sets_128
python predict.py --experiment dkvb_all_sets_128

python train.py --experiment vq_all_sets_512
python predict.py --experiment vq_all_sets_512

python train.py --experiment dkvb_all_sets_512
python predict.py --experiment dkvb_all_sets_512
