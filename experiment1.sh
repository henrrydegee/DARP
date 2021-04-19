#!/bin/bash

# Variables
FOLDERNAME="result5"
SETTINGTXT="${FOLDERNAME}/details.txt"
PROGRESSTXT="${FOLDERNAME}/progress.txt"

# Experiment Settings
EPOCHS=500
START=0
BATCH=64
LEARNING_RATE=0.002

# Method Options (Class Dataset & Model Settings)
NUM_MAX=1500
RATIO=2.0
IMB_L=100
IMB_U=100
VAL_ITER=500
NUM_VAL=10

# Hyperparameters for FixMatch
TAU=0.95
EMA_DECAY=0.999
LAMBDA_U=1

# Hyperparameters for FixMatch
WARM=200
ALPHA=2.0
DARP="" # "--darp" to use DARP, "" to NOT use DARP
EST="" # "--est" to use Estimated Distribution for Unlabeled Dataset
        # "" to NOT use Estimated Distribution for Unlabeled Dataset
ITER_T=10
NUM_ITER=10

# Settings Used for Weighted Loss based on Class Distribution
W_L="total"
DISTB="pseudo"

# For More Info: Execute "python3 train_fix.py --help"

mkdir $FOLDERNAME
# Print Settings for Reference
echo -e \
"Settings Used for $FOLDERNAME @ $(pwd) : \n \
\n \
Total Epochs  = $EPOCHS \n \
Start Epoch   = $START \n \
Batch Size    = $BATCH # (for each Supervised & Unsupervised) \n \
Learning Rate = $LEARNING_RATE # Learning Rate \n \
\n \
# Method Options (Class Dataset & Model Settings) \n \
Number of Samples in Maximal Class = $NUM_MAX \n \
Labeled vs Unlabeled Data Ratio    = $RATIO \n \
Imbalance Ratio for Labeled Data   = $IMB_L \n \
Imbalance Ratio for Unlabeled Data = $IMB_U \n \
Frequency of Evaluation            = $VAL_ITER \n \
Number of Validation Data          = $NUM_VAL \n \
\n \
# Hyperparameters for FixMatch \n \
Minimal Confidence for Pseudo-Label (tau) = $TAU \n \
EMA Decay Hyperparameter                  = $EMA_DECAY \n \
Weight for Unsupervised Loss (Lambda_u)   = $LAMBDA_U \n \
\n \
# Hyperparameters for DARP \n \
Warmup Epoch                              = $WARM \n \
Hyperparameter for removing Noisy Entries = $ALPHA \n " > $SETTINGTXT

if [ "$DARP" = "" ] ; then
    echo -e "Using DARP = False" >> $SETTINGTXT
elif [ "$DARP" = "--darp" ] ; then
    echo -e "Using DARP = True"  >> $SETTINGTXT
else
    echo -e "Error in DARP Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$EST" = "" ] ; then
    echo -e "Using Estimated Distribution for Unlabeled Dataset = False" >> $SETTINGTXT
elif [ "$EST" = "--est" ] ; then
    echo -e "Using Estimated Distribution for Unlabeled Dataset = True"  >> $SETTINGTXT
else
    echo -e "Error in EST Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

echo -e \
"Number of iteration (T) for DARP          = $ITER_T \n \
Scheduling for updating Pseudo-Labels      = $NUM_ITER \n \
\n \
# Settings Used for Weighted Loss based on Class Distribution \n" >> $SETTINGTXT

if [ "$W_L" = "total" ] ; then
    echo -e "Weight Loss Formula Used (Based on Sum Distribution): \n \
    class_weight_u = distb_u / torch.sum(distb_u) * 2 + 1 \n \
    Note: class_weight_u = [1, 3] \n" >> $SETTINGTXT
elif [ "$W_L" = "minority" ] ; then
    echo -e "Weight Loss Formula Used (Based on Minority Distribution): \n \
    class_weight_u = (distb_u / lowest_ref) ** (1/3) \n \
    Note: class_weight_u = [1, inf] \n"  >> $SETTINGTXT
elif [ "$W_L" = "" || "$DISTB" = "" ] ; then
    echo -e "Using Equal Weighting (torch.Ones(num_class))" >> $SETTINGTXT
else
    echo -e "Error in W_L Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

if [ "$DISTB" = "pseudo" ] ; then
    echo -e "Class Distribution used  = Psuedo-Label (p_hat)" >> $SETTINGTXT
elif [ "$DISTB" = "output" ] ; then
    echo -e "Class Distribution used  = Model Prediction (q / y_hat_u)"  >> $SETTINGTXT
elif [ "$DISTB" = "" || $W_L = "" ] ; then
    echo -e "No Class Distribution was used" >> $SETTINGTXT
else
    echo -e "Error in DISTB Setting, Please Correct it. Exiting..." \
    | tee -a $SETTINGTXT
    exit 1
fi

# Execute Experiment
python3 train_fix.py \
--epochs $EPOCHS \
--start-epoch $START \
--batch-size $BATCH \
--lr $LEARNING_RATE \
\
--num_max $NUM_MAX \
--ratio $RATIO \
--imb_ratio_l $IMB_L \
--imb_ratio_u $IMB_U \
--val-iteration $VAL_ITER \
--num_val $NUM_VAL \
\
--tau $TAU \
--ema-decay $EMA_DECAY \
--lambda_u $LAMBDA_U \
\
--warm $WARM \
--alpha $ALPHA \
$DARP \
$EST \
--iter_T $ITER_T \
--num_iter $NUM_ITER \
\
--w_L $W_L \
--distb $DISTB \
\
--out $FOLDERNAME \
\
| tee $PROGRESSTXT

# Note: Use tee -a to append text
# End of Experiment