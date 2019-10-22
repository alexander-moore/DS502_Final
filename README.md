# DS502_Final
DS502 Final Project

Outline Submission due Oct 29

https://www.kaggle.com/c/nfl-big-data-bowl-2020

## Problem Description
This problem is about football

## Data Description
Take from Kaggle and cite

## Prediction type
IT COULD BE FRAMED AS A REGRESSION OR CLASSIFICATION PROBLEM. WE ARE ESTIMATING THE CUMULATIVE DISTRIBUTION FUNCTION, SO WE COULD EITHER PREDICT <5,  OR WE COULD DO THE CLASSIFICAITON:
[0 0 0 0 0 1 1 1 ]
OR
[0 0 0 0 0 1 0 0 ]

TONS OF REALLY GOOD DISCUSSIONS ON THE KAGGLE PAGE ABOUT THIS

## Methods
We can do some neural networks, but the point of our analysis can be how different models perform and why. We could say like, 4 tweaks on model X, or just 4 models

## Error Metrics and Algorithms for assessing them
The kaggle competition stipulates an error metric based on the cumulative distribution of yards gained.

## Comments and Concerns
The NFL data has an interesting structure which gives us unique insights and challenges. Rather than each row being an observation, collections of rows unified by a 'Play I.D.' represent one play during a football game, with a response yards gained. There are many interesting preprocessing techniques to transform this data into something we can model. There are countless exploration topics in the data on topics such as matchups, player performance, and team organization that can motivate our models and insights about this unique data.

The high variability in the data is one concern for our regression. An identical play (players, stadium, latent factors) could lead to a 0 yard gain or a 100-yard touchdown. For this reason we will have to negotiate with extreme noise in the developement of our model, and in some cases a lack of predictor correlation to the response.


![do not click](https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.catit.com%2Fwp-content%2Fuploads%2F2017%2F05%2FAnne-Baukje-Oord-891x891.jpg&f=1&nofb=1)
