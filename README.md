# weather_aware_pipeline

#Project: Weather-Aware Sales Forecasting Pipeline			                            		     Oct 2023 – Dec 2023
Implemented a Weather-Aware Sales Forecasting Pipeline to analyze the impact of climate conditions on retail sales across six high-priority SKUs in Massachusetts and Arizona. This cross-functional initiative integrated 3 years of internal transactional data from AWS RDS with external NOAA weather data in Amazon S3, forming a unified dataset of 100,000+ records using AWS Glue. The project developed predictive models on Amazon SageMaker to forecast sales, revealing critical region-specific trends (e.g., rainfall decreased AZ sales by 18%; cooler temperatures boosted MA sales by 23%). Insights delivered through Tableau and PowerPoint optimized store-level stocking strategies and enhanced inventory management.
#Roles and Responsibilities:
•	Queried 3 years of transactional data from AWS RDS (MySQL) and merged it with NOAA-sourced weather data stored in Amazon S3, creating a unified dataset of 100,000+ records using Python (pandas) and SQL.
•	Standardized, cleaned, and enriched data via AWS Glue jobs and Jupyter notebooks running on Amazon SageMaker; handled missing values, normalized features, and engineered new weather impact variables (e.g., moving averages, lag features).
•	Performed exploratory analysis to uncover correlations between SKU sales and weather variables (e.g., temperature, precipitation), revealing region-specific trends that influenced in-store foot traffic and product demand.
•	Developed predictive models(Random Forest and Linear Regression) using scikit-learn on SageMaker to forecast sales.
•	Delivered prescriptive insights that informed store-level stocking strategies
•	Visualized insights via dashboards using Tableau, and presented findings to stakeholders with actionable recommendations using Tableau and PowerPoint.
