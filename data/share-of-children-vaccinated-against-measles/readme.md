# Share of one-year-olds vaccinated against measles - Data package

This data package contains the data that powers the chart ["Share of one-year-olds vaccinated against measles"](https://ourworldindata.org/grapher/share-of-children-vaccinated-against-measles?v=1&csvType=full&useColumnShortNames=false) on the Our World in Data website. It was downloaded on September 09, 2025.

### Active Filters

A filtered subset of the full data was downloaded. The following filters were applied:

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## Share of one-year-olds who have had one dose of the measles vaccine
Share of one-year-olds who have had the first dose of the measles vaccine in a given year.
Last updated: July 15, 2025  
Next update: July 2026  
Date range: 1980–2024  
Unit: %  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
WHO & UNICEF (2025); UN, World Population Prospects (2024) – processed by Our World in Data

#### Full citation
WHO & UNICEF (2025); UN, World Population Prospects (2024) – processed by Our World in Data. “Share of one-year-olds who have had one dose of the measles vaccine” [dataset]. WHO & UNICEF, “WHO Immunization Data - Vaccination coverage”; United Nations, “World Population Prospects” [original data].
Source: WHO & UNICEF (2025), UN, World Population Prospects (2024) – processed by Our World In Data

### What you should know about this data
* Measles is one of the most contagious diseases. The first dose of measles vaccine is critical for building immunity. In countries where the national schedule recommends the first dose at 12 months or later based on local epidemiology, these estimates reflect the percentage of children who received their first dose as recommended.
* This chart shows official estimates of national immunization coverage published by the WHO and UNICEF. The estimates include all WHO member states, even those that did not report 2023 data. For non-reporting countries, WHO uses statistical methods to extrapolate from previously reported data, ensuring global coverage can be assessed.
* Global and regional vaccination coverage is calculated using population-weighted averages. In 2023, approximately 5% of countries did not report data, requiring extrapolation from their 2022 data to maintain complete global estimates.
* These estimates combine several sources: official administrative data from health facilities, coverage surveys that meet WHO quality standards, and other relevant information like vaccine supply issues or schedule changes. The accuracy of these estimates depends on how complete and reliable each country’s reporting systems are.

### Sources

#### WHO & UNICEF – WHO Immunization Data - Vaccination coverage
Retrieved on: 2025-07-15  
Retrieved from: https://immunizationdata.who.int/global?topic=Vaccination-coverage&location=  

#### United Nations – World Population Prospects
Retrieved on: 2024-07-11  
Retrieved from: https://population.un.org/wpp/downloads/  


    