# The homework3 of data mining
## Decision Trees model
### Learning decision trees, entropy as criteria, max depth allowed 11
```sh
python -m hw3.models.runModels --dataset connect-4 --train_test_split 0.2 --model dtree --seed 12 --max_depth 11 --splitting_criteria entropy
```
### Learning decision trees, Gini as criteria, max depth allowed 11
```sh
python -m hw3.models.runModels --dataset connect-4 --train_test_split 0.2 --model dtree --seed 12 --max_depth 11 --splitting_criteria gini
```
### Deep Neural Network
### Learning neural network, learning rate of 0.05, batch size of 1 and logging accuracy and loss at every 5 epochs for 50 epochs with seed of 12
```sh
python -m hw3.models.runModels --dataset connect-4 --train_test_split 0.2 --model nnet --log_interval 5 --batch_size 1 --epochs 50 --lr 0.05 --seed 12
```
### SVM
```sh
python -m hw3.models.runModels --dataset connect-4 --model svm
```
### Logistic Regression
```sh
python -m hw3.models.runModels --dataset connect-4 --model lr
```
### Random Forest
```sh
python -m hw3.models.runModels --dataset connect-4 --model rf
```
