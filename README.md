# Predicting at-risk gamers using their bio/social profiles
This model aims to predict gamers more likely to develop negative emotions such as anxiety, restleness, irritability and so on, using their bio and social profiles.

<p>The data was collected as part of a study by Marian Sauter and Dejan Draschkow and is available for public download <a href="https://osf.io/vnbxk/">here</a>. The sample size of participants was ~13000. </p>

<p>An exploratory data analysis revealed some tendencies between negative feelings and a few of the data points :</p>

<p><b>1. Number of hours played :</b></p>

![](images/Hours.png)

<p><b>2. Reason for playing :</b></p>

![](images/Reason.png)

<p><b>3. Age group :</b></p>

![](images/Age.png)

<p><b>4. Employment Status :</b></p>

![](images/Work.png)

<p><b>5. Education :</b></p>

![](images/Degree.png)

<p><b>6. Playing Environment :</b></p>

![](images/PlayStyle.png)

However, no feature showed any great correlation with %risk:

![](images/correlation.png)

As a result, a logistic regression model did not show great accuracy in predicting at-risk players based on their bio-social profiles. Some adjustments such as cutting age and hours played into bins did not improve the accuracy.





