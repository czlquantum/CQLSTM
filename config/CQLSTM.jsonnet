
local batch_size = 32;
local cuda_device = 0;
local num_epochs = 2;
local seed = 42;

local embedding_dim = 768;
local output_dim = 128;
local dropout = 0.2;
local lr = 0.00003;
local output_dim = 128;
local model_name = 'RoBERTa';
local ptm_name = 
  if model_name == 'RoBERTa'
  then 'roberta-base'
  else 'bert-base-cased';


local data_dir = './data/';
local get_train_path =  data_dir + task_name + '/' + task_name + '_train.txt';
local get_val_path =  data_dir + task_name + '/' + task_name + '_test.txt';

// Please choose dataset with task_name! ['CR', 'MPQA', 'MR', 'SUBJ']
local task_name = 'CR';
local num_classes = 2;

local train_path =  get_train_path(task_name);
local val_path = get_val_path(task_name);

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  
  dataset_reader: {
    type: 'my_dataset_reader',

    tokenizer: {
      type: 'pretrained_transformer',
      model_name: ptm_name
    },

    token_indexers: {
      tokens: {
        type: 'pretrained_transformer',
        model_name: ptm_name
      },
    },
    task_name: task_name,
  },

  datasets_for_vocab_creation: ['train'],
  train_data_path: train_path,
  validation_data_path: val_path,
  
  model: {
    type: 'complex_text_classifier',
    embedder_real: {
      token_embedders: {
        tokens: {
          type: 'pretrained_transformer',
          model_name: ptm_name
        },
      },
    },
    embedder_imag: {
      token_embedders: {
        tokens: {
          type: 'pretrained_transformer',
          model_name: ptm_name
        },
      },
    },
    encoder: {
      type: 'CQLSTM',
      embedding_dim: embedding_dim,
      output_dim: output_dim
    },
    pooler: {
      type: 'bert_pooler',
      pretrained_model: ptm_name
    },
    num_classes: num_classes
  },
  data_loader: {
    shuffle: true,
    batch_size: batch_size,
  },
  trainer: {
    cuda_device: cuda_device,
    num_epochs: num_epochs,
    optimizer: {
      lr: lr,
      // "type": "huggingface_adamw",
      type: 'adamw',
    },
    validation_metric: '+fscore',
  }
}
