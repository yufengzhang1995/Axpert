NUM_EPOCHS = 4        #Number of epochs to train for
BATCH_SIZE = 8      #Change this depending on GPU memory
NUM_WORKERS = 1      #A value of 0 means the main process loads the data
LEARNING_RATE = 2e-5
LOG_EVERY = 20      #iterations after which to log status during training test: 10
VALID_NITER = 20   #iterations after which to evaluate model and possibly save (if dev performance is a new max) test: 5
PRETRAIN_PATH = None  #path to pretrained model, such as BlueBERT or BioBERT
PAD_IDX = 0           #padding index as required by the tokenizer 
WEIGHTED_LOSS = [1,5, 5]
#CONDITIONS is a list of all medical observations 
CONDITIONS = ['pneumatosis', 'pvg','freeair']
CLASS_MAPPING = {0: "Negative", 1: "Positive", 2: "Uncertain"}