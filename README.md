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

### :small_blue_diamond: without hidden layers

### :small_blue_diamond: with 1 hidden layer

### :small_blue_diamond: with 2 hidden layers

### :small_blue_diamond: with 3 hidden layers

## :small_blue_diamond: Conclusions


