# :pill: Predicting diabetes

## :pushpin: Purpose of the work
In this work, I want to build a neural network that can best predict whether a person has diabetes or not.

## :bar_chart: EDA - Exploratory data analysis
| Name                   | Type    | Description                                                                                                           | Missing Values |
|------------------------|---------|-----------------------------------------------------------------------------------------------------------------------|----------------|
| ID                     | Integer | Patient ID                                                                                                            | No             |
| Diabetes_binary        | Binary  | 0 = no diabetes, 1 = prediabetes or diabetes                                                                          | No             |
| HighBP                 | Binary  | 0 = no high BP, 1 = high BP                                                                                           | No             |
| HighChol               | Binary  | 0 = no high cholesterol, 1 = high cholesterol                                                                         | No             |
| CholCheck              | Binary  | 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years                                             | No             |
| BMI                    | Integer | Body Mass Index                                                                                                       | No             |
| Smoker                 | Binary  | Have you smoked at least 100 cigarettes in your entire life? (5 packs = 100 cigarettes) 0 = no, 1 = yes               | No             |
| Stroke                 | Binary  | (Ever told) you had a stroke. 0 = no, 1 = yes                                                                         | No             |
| HeartDiseaseorAttack   | Binary  | Coronary heart disease (CHD) or myocardial infarction (MI) 0 = no, 1 = yes                                            | No             |
| PhysActivity           | Binary  | Physical activity in past 30 days (not including job) 0 = no, 1 = yes                                                 | No             |
| Fruits                 | Binary  | Consume fruit 1 or more times per day 0 = no, 1 = yes                                                                 | No             |
| Veggies                | Binary  | Consume vegetables 1 or more times per day 0 = no, 1 = yes                                                            | No             |
| HvyAlcoholConsump      | Binary  | Heavy drinkers (Men > 14 drinks/week, Women > 7 drinks/week) 0 = no, 1 = yes                                          | No             |
| AnyHealthcare          | Binary  | Have any kind of health care coverage (insurance, HMO, etc.) 0 = no, 1 = yes                                          | No             |
| NoDocbcCost            | Binary  | Could not see a doctor due to cost in the past 12 months? 0 = no, 1 = yes                                             | No             |
| GenHlth                | Integer | General health rating (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor)                                    | No             |
| MentHlth               | Integer | Days in past 30 when mental health was not good (1-30 days)                                                           | No             |
| PhysHlth               | Integer | Days in past 30 when physical health was not good (1-30 days)                                                         | No             |
| DiffWalk               | Binary  | Serious difficulty walking or climbing stairs? 0 = no, 1 = yes                                                        | No             |
| Sex                    | Binary  | 0 = female, 1 = male                                                                                                  | No             |
| Age                    | Integer | 13-level age category (1 = 18-24, 9 = 60-64, 13 = 80 or older)                                                        | No             |
| Education              | Integer | Education level (1 = No school, 2 = Elementary, 3 = Some HS, 4 = HS graduate, 5 = Some college, 6 = College graduate) | No             |
| Income                 | Integer | Income scale (1 = < $35,000, 8 = ≥ $75,000)                                                                           | No             |

Because there is many binary features I normalize the rest of the variables to have values between 0 and 1.

After examining the boxplot, histograms and basic statistics, the following conclusions can be drawn about the dataset:
- most people rated their health as good or better
- at least half rated their mental and physical health as good every day in the past 30 days
- mean age is between 55 and 59, around 25% people is below 35 years old
- at least half the people attended college
- at least 25% of the people earn $75,000
- mean value of the BMI is euqal 28.38 and this is overveight,
- according to the standard definitions, a person is considered to have a normal weight if their BMI is below 24.9, and they are in the first quartile, which is defined as a BMI of 24. This indicates that slightly more than 25% of the population has a normal weight or is underweight.
- the boxplot shows that there is several people with BMI above 40 - this mean Obesity class III (the last group according BMI classification)
- there are more people with lower blood pressure and cholesterol levels,
- most people have had their cholesterol checked in the past five years,
- there are slightly more non-smokers,
- people who have had a stroke or heart attack are a small percentage of the group,
- almost 4 times as many people have done physical activity outside of work in the last 30 days and the number of people who eat at least one vegetable a day is also similar;
- it's interesting that more people eat vegetables every day than fruit.
- the vast majority do not drink alcohol often and have health insurance.
- a small percentage of people needed medical assistance but were unable to seek it due to the cost.
- most people had no problems climbing stairs or taking demanding walks.
- the data includes slightly more women than men.

> [!IMPORTANT]
> Most of the people in this group are not diabetic. This is important when we score the model because if the model will classify all people as healthy, it will still have good accuracy (in this case 86%). So, it's also important to look at precision and recall.

## :small_blue_diamond: Neural networks

Each network model was trained on different combinations of 3 parameters:
- batch size: 32, 64, 128, 256;
- learning rate: 0.1, 0.01, 0.001, 0.0001;
- activation function: Sigmoid, ReLU, Tanh, Softplus.

Pozostałe parametry zawsze były stałe:
- number of epochs = 8
- loss function = Binary Cross Entropy
- optimization function = Adam()

### :small_blue_diamond: without hidden layers

```python
class NeuralNetwork1layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
```

| Learning Rate | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|--------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 1e-04       | 83.29        | 0.56                       | 0.38          | 0.03                        | 0.33       | 0.03                     |
| 0.001       | 83.50        | 0.36                       | 0.41          | 0.00                        | 0.41       | 0.04                     |
| 0.01        | 84.68        | 0.67                       | 0.44          | 0.02                        | 0.36       | 0.06                     |
| 0.1         | 84.49        | 2.27                       | 0.47          | 0.09                        | 0.23       | 0.22                     |

| Batch Size | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 32         | 83.94        | 0.75                       | 0.41          | 0.02                        | 0.35       | 0.07                     |
| 64         | 83.55        | 1.80                       | 0.41          | 0.04                        | 0.39       | 0.11                     |
| 128        | 84.21        | 1.32                       | 0.45          | 0.09                        | 0.30       | 0.20                     |
| 256        | 84.27        | 1.43                       | 0.42          | 0.06                        | 0.29       | 0.11                     |

The greater the amount of data in the sample, the better the results, but the variances are also high, meaning that results can be weaker than the average.


### :small_blue_diamond: with 1 hidden layer

```python
class NeuralNetwork2layers(nn.Module):
    # dzięki parametrom można później ustalić ilość neuronów w warstwie oraz wybrać funkcję aktywacji
    def __init__(self, features=16, activation=nn.ReLU()):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, features),
            activation,
            nn.Linear(features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
```

| Learning Rate | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|--------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 1e-04       | 83.62        | 0.54                       | 0.42          | 0.01                        | 0.44       | 0.04                     |
| 0.001       | 83.45        | 1.23                       | 0.42          | 0.02                        | 0.49       | 0.06                     |
| 0.01        | 84.18        | 1.51                       | 0.44          | 0.04                        | 0.41       | 0.11                     |
| 0.1         | 82.75        | 5.36                       | 0.13          | 0.17                        | 0.24       | 0.32                     |

| Activation Function | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|--------------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| ReLU             | 82.84        | 3.87                       | 0.36          | 0.15                        | 0.41       | 0.21                     |
| Sigmoid         | 83.23        | 3.59                       | 0.34          | 0.17                        | 0.40       | 0.22                     |
| Softplus        | 84.25        | 1.45                       | 0.38          | 0.15                        | 0.39       | 0.17                     |
| Tanh            | 83.67        | 1.67                       | 0.33          | 0.17                        | 0.38       | 0.20                     |

| Batch Size | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 32         | 84.46        | 1.55                       | 0.33          | 0.20                        | 0.32       | 0.21                     |
| 64         | 84.09        | 1.62                       | 0.34          | 0.17                        | 0.37       | 0.19                     |
| 128        | 82.00        | 4.75                       | 0.38          | 0.12                        | 0.47       | 0.19                     |
| 256        | 83.44        | 1.62                       | 0.36          | 0.14                        | 0.41       | 0.17                     |

Looking already at the exact statistics, the network achieves the best results with a learning rate of 0.01, a softplus activation function and a batch size of 32. With a small learning rate, the results are similar regardless of the other parameters, while for a rate of 0.1, the other parameters already have a strong influence on the results.


### :small_blue_diamond: with 2 hidden layers

```python
class NeuralNetwork3layers(nn.Module):
    def __init__(self, features1=16, features2=16, activation=nn.ReLU()):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, features1),
            activation,
            nn.Linear(features1, features2),
            activation,
            nn.Linear(features2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
```

| Learning Rate | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|--------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 1e-04       | 83.39        | 0.91                       | 0.42          | 0.02                        | 0.47       | 0.04                     |
| 0.001       | 83.64        | 0.88                       | 0.43          | 0.02                        | 0.48       | 0.05                     |
| 0.01        | 83.94        | 1.81                       | 0.40          | 0.11                        | 0.39       | 0.18                     |
| 0.1         | 84.48        | 3.60                       | 0.07          | 0.14                        | 0.12       | 0.27                     |

| Activation Function | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|--------------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| ReLU             | 83.11        | 2.72                       | 0.36          | 0.15                        | 0.42       | 0.19                     |
| Sigmoid         | 84.14        | 1.54                       | 0.32          | 0.19                        | 0.36       | 0.22                     |
| Softplus        | 84.32        | 2.07                       | 0.35          | 0.18                        | 0.36       | 0.20                     |
| Tanh            | 83.87        | 1.90                       | 0.27          | 0.19                        | 0.33       | 0.26                     |

| Batch Size | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 32         | 84.40        | 1.86                       | 0.30          | 0.21                        | 0.31       | 0.23                     |
| 64         | 84.53        | 1.25                       | 0.32          | 0.19                        | 0.32       | 0.22                     |
| 128        | 83.34        | 2.72                       | 0.33          | 0.17                        | 0.40       | 0.21                     |
| 256        | 83.17        | 2.14                       | 0.36          | 0.14                        | 0.43       | 0.19                     |

As before, networks with a learning rate of 0.1 have a high deviation for the test data and, of all the activation functions, the ReLU function performs worst for the test set.

### :small_blue_diamond: with 3 hidden layers

```python
class NeuralNetwork4layers(nn.Module):
    def __init__(self, features1=16, features2=16, features3=16, activation=nn.ReLU()):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, features1),
            activation,
            nn.Linear(features1, features2),
            activation,
            nn.Linear(features2, features3),
            activation,
            nn.Linear(features3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
```

| Learning Rate | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|--------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 1e-04       | 83.28        | 1.15                       | 0.42          | 0.02                        | 0.48       | 0.06                     |
| 0.001       | 83.20        | 1.49                       | 0.42          | 0.02                        | 0.50       | 0.06                     |
| 0.01        | 84.19        | 1.61                       | 0.37          | 0.15                        | 0.36       | 0.20                     |
| 0.1         | 81.41        | 18.00                      | 0.03          | 0.11                        | 0.09       | 0.27                     |

| Activation Function | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|--------------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| ReLU             | 79.95        | 17.64                      | 0.36          | 0.16                        | 0.42       | 0.23                     |
| Sigmoid         | 84.02        | 1.37                       | 0.31          | 0.19                        | 0.36       | 0.22                     |
| Softplus        | 84.14        | 1.54                       | 0.32          | 0.19                        | 0.36       | 0.22                     |
| Tanh            | 83.97        | 2.37                       | 0.25          | 0.20                        | 0.30       | 0.27                     |

| Batch Size | Mean Accuracy | Standard Deviation Accuracy | Mean Precision | Standard Deviation Precision | Mean Recall | Standard Deviation Recall |
|------------|--------------|----------------------------|---------------|-----------------------------|------------|--------------------------|
| 32         | 84.62        | 1.46                       | 0.30          | 0.21                        | 0.30       | 0.22                     |
| 64         | 79.44        | 17.58                      | 0.29          | 0.19                        | 0.40       | 0.28                     |
| 128        | 84.11        | 1.59                       | 0.34          | 0.17                        | 0.36       | 0.22                     |
| 256        | 83.91        | 1.55                       | 0.31          | 0.19                        | 0.37       | 0.23                     |


The example of a network with three layers shows even better the conclusions observed
in the previous examples.

## :small_blue_diamond: Conclusions

| Number of hidden layers               | 0  | 1  | 2  | 3  |
|---------------------------------------|----|----|----|----|
| **Mean accuracy (training)**       | 83 | 83 | 84 | 83 |
| **Mean accuracy (test)**          | 84 | 84 | 84 | 83 |
| **Mean precision (training)**         | 39 | 38 | 35 | 32 |
| **Mean precision (test)**            | 42 | 35 | 32 | 31 |
| **Mean recall (training)**          | 38 | 40 | 38 | 36 |
| **Mean recall (test)**             | 33 | 39 | 37 | 36 |

The number of layers does not have a large impact on the accuracy of the network, in each case the accuracy is around 83% - 84%.In the case of precision and sensitivity, better results are achieved by networks
with fewer hidden layers.
In summary, the learning rate and the activation function have the greatest influence on the results, but in most cases the accuracy is around 83% with precision
and sensitivity varying between 30% and 40%.




