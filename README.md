

**Dependencies**

> 1. pytorch>=1.12
> 2. allennlp==2.10
> 3. complexPyTorch=0.4
>


## Some notes about allennlp
1. --dry-run  # load dataset but do not train the model
2. -f  # force training, this command will override the save path
3. -s  # save path
4. --include-package $path  # personal work path including model, classifier and so on
5. config/xxx.jsonnet  # config file with jsonnet format


## Our train command

### CQLSTM
```cmd
allennlp train config/CQLSTM.jsonnet --include-package work -s ./result/cr_CQLSTM -f
```


### QIM
```cmd
allennlp train config/QIM.jsonnet --include-package work -s ./result/cr_QIM -f
```

## Dataset
Please modify the variable "task_name" in model.jsonnet to change different datasets.

## Other command
```cmd
# delete training models
rm ./result/*/*_state_*.th -f
```