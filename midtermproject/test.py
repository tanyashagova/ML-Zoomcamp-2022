import requests
url = 'http://localhost:8080/predict'

record = {'Age': 33, 'Attrition': 1, 
        'BusinessTravel': 'travel_frequently', 'DailyRate': 1076, 
        'Department': 'research_development', 'DistanceFromHome': 3, 
        'Education': 'bachelor', 'EducationField': 'life_sciences',
        'EmployeeCount': 1, 'EmployeeNumber': 702, 
        'EnvironmentSatisfaction': 'low', 'Gender': 'male', 'HourlyRate': 70, 
        'JobInvolvement': 'high', 'JobLevel': '1l', 'JobRole': 'research_scientist', 
        'JobSatisfaction': 'low', 'MaritalStatus': 'single', 'MonthlyIncome': 3348, 
        'MonthlyRate': 3164, 'NumCompaniesWorked': 1, 'Over18': 'y', 'OverTime': 'yes', 
        'PercentSalaryHike': 11, 'PerformanceRating': 'excellent', 
        'RelationshipSatisfaction': 'low', 'StandardHours': 80, 'StockOptionLevel': '0l', 
        'TotalWorkingYears': 10, 'TrainingTimesLastYear': 3, 'WorkLifeBalance': 'better', 
        'YearsAtCompany': 10, 'YearsInCurrentRole': 8, 'YearsSinceLastPromotion': 9,
        'YearsWithCurrManager': 7}

result = requests.post(url, json=record).json()
print(result)