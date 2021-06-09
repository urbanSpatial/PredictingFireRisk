
<br>

<center>
<p style="line-height:1">
 <font size="30"> Geospatial Machine Learning: Predicting fire risk in San Francisco</font> 
</p>
</center>

<br>

<center><font size="5"> Ken Steif, Ph.D</font></center> 

<center>

<a href="https://twitter.com/KenSteif">@KenSteif</a>

http://UrbanSpatialAnalysis.com/

</center> 

<center>
![](C:/Users/ksteif/Desktop/Spring_2021/predictingFires/predictinig fires urban spatial logo.png){width=30% height=30%}
</center>

<br>

# 1. Introduction 

This markdown is one component of a workshop created for the University Consortium for GIS, 2021 Symposium. We will learn how to develop a geospatial risk prediction model - a type of machine learning model, specifically to predict fire risk in San Francisco, California. 

This workshop is an abridged version of Chapter 5 on Predictive Policing found in a new book, [Public Policy Analytics](https://urbanspatial.github.io/PublicPolicyAnalytics/index.html) by Ken Steif. The book is published by CRC Press, but is also available as open source.

Here we posit that fire risk across space is a function of exposure to nearby spatial risk or protective factors, like blight or aging infrastructure. These algorithms are explicitly spatial in nature because of their reliance on measuring geographic exposure. 

These models provide intelligence beyond points on a map - that simply tell us to revisit past locations of adverse events. They provide a more thorough predictive understanding of spatial systems and why things occur where.

Fire is a relatively rare event, and the 'latent risk' for fire is much greater than the number of fires actually observed. Today, we will estimate this spatial latent risk to help the San Francisco Fire Department do operational planning, like fire inspection and smoke detector outreach.

Please check out the accompanying video for a lot more context on fire prediction and algorithmic bias. Minimal context is provided here. 

This tutorial assumes the reader is familiar with spatial analysis and data wrangling in R - namely the `sf` and `tidyverse` packages. It is also assumed the reader has some familiarity with machine learning.


# 2. Data wrangling

In this section, we download spatial datasets from the San Francisco open data [website](https://datasf.org/) on building fires, neighborhoods, land use, vacancy, graffiti and more. We then learn how to measure geographic exposure by compiling these data into the `fishnet`, a lattice grid of cells that is the unit of analysis for the algorithm. 

Begin by installing and loading the below libraries. The `source` loads a set of functions from github, some of which are used in this workshop. 

```r
options(scipen=9999)

library(tidyverse)
library(sf)
library(RSocrata)
library(spatstat)
library(viridis)
library(FNN)
library(spdep)
library(gridExtra)

source("https://raw.githubusercontent.com/urbanSpatial/Public-Policy-Analytics-Landing/master/functions.r")
```

## Download & wrangle fire data

We begin by downloading basemap layers and creating the `fishnet` using 500ft by 500ft grid cells. Note that all data is projected as a State Plane (feet). `main_streets` are filtered out for our basemap.

The resulting `fishnet` is created with `st_make_grid` and visualized below. A `uniqueID` is created for joining below.

```r 
nhoods <- 
  st_read("https://data.sfgov.org/api/geospatial/p5b7-5n3h?method=export&format=GeoJSON") %>%
  st_transform('ESRI:102642') %>%
  st_buffer(-1) #to correct some of the digitization errors
  
bound <- 
  st_read("https://opendata.arcgis.com/datasets/4d9f9059e4da4c4e83b7d56f62d0f8aa_0.geojson") %>%
  st_transform(st_crs(nhoods)) %>%
  filter(OBJECTID == 467)

str <- st_read("https://data.sfgov.org/api/geospatial/3psu-pn9h?method=export&format=GeoJSON")
main_streets <- filter(str, street %in% c("MARKET","PORTOLA"))

fishnet <- 
  st_make_grid(nhoods, cellsize = 500) %>%
  st_sf() %>%
  mutate(uniqueID = rownames(.)) %>%
  .[bound,] #return only grid cells in the San Fran boundary

ggplot() + 
  geom_sf(data=fishnet) + 
  geom_sf(data=bound, fill=NA) +
  labs(title="San Francisco & Fishnet") +
  mapTheme()
```

Next, building fires are downloaded using the `read.socrata` function. Socrata is the open data platform used by San Francisco - and the `RSocrata` package enables quick data downloads. Note the `?` operator in the query, which returns only fires labeled as building fires. 

We `filter` for fires after 2016, convert the data to the sf format (`st_as_sf`) and then project into feet. It is clear that fires occur across the City with a higher density in the Tenderloin neighborhood to the northeast.

```r
fires <- 
  read.socrata("https://data.sfgov.org/Public-Safety/Fire-Incidents/wr8u-xric?Primary Situation=111 Building fire") %>%
    mutate(year = substr(incident_date,1,4)) %>% 
    filter(year >= 2017) %>%
    st_as_sf(wkt="point") %>%
    st_set_crs(4326) %>%
    st_transform(st_crs(nhoods)) %>%
    dplyr::select(year) %>%
    .[nhoods,]
    
ggplot() + 
  geom_sf(data=nhoods) + 
  geom_sf(data=main_streets, colour="white", size=2) +
  geom_sf(data=fires) +
  labs(title="Building fires, San Francisco: 2017 - 2021",
       subtitle="Market Street in white") +
  mapTheme()
```

Our predictive model is going to be 'trained' on fires from 2017 to 2020, and below, 2021 fires are set aside for testing purposes. If the goal is operational planning, then we should ensure our model generalizes to the the near future - hence the 2021 fires are set aside. Besides, it is unimpressive to predict for the data the model was trained on.

```r 
fires17_20 <- filter(fires, year < 2021)
fires21 <- filter(fires, year == 2021)
```

How do we know if our predictions are useful? In this case, we will compare them to a Kernel Density map of fires - which we can think of as predictions generated purely from spatial autocorrelation (the idea that fires cluster). This is different from a geospatial risk prediction model which will include some spatial autocorrelation predictors, but relies on other measures of geographic exposure - like blight.

The first two lines in the code block below use the `spatstat` package to create the Kernel Density layer^[Depending on your `spatstat` version, you may have to remove `spatstat::` from line two in this code block]. The third line converts it to a data frame, then a Simple Features layer, before joining it to the `fishnet`. The `aggregate` function spatial joins the Density layer and the `fishnet` to return the mean density per grid cell.

```r
fire_ppp <- as.ppp(st_coordinates(fires17_20), W = st_bbox(nhoods))
fire_KD <- spatstat::density.ppp(fire_ppp, 1000)

as.data.frame(fire_KD) %>%
  st_as_sf(coords = c("x", "y"), crs = st_crs(nhoods)) %>%
  aggregate(., fishnet, mean) %>%
   ggplot() +
     geom_sf(aes(fill=value)) +
     geom_sf(data=main_streets, colour="white", size=1) +
     scale_fill_viridis(name = "Density",
                        option = "magma") +
     labs(title = "Kernel density risk of fires",
          subtitle = "Market St. overlayed") +
     mapTheme()    
```

The code block below spatially joins the 2017-2019 building fire points to the `fishnet`. `aggregate` is again used to perform a spatial join. Any grid cell without a fire returns `NA` which is replaced with `0`. The result is mapped. Now we have our dependent variable - the count of fires per grid cell.

```r
fire_net <- 
  dplyr::select(fires17_20) %>% 
  mutate(countFires = 1) %>% 
  aggregate(., fishnet, sum) %>%
  mutate(countFires = replace_na(countFires, 0),
         uniqueID = rownames(.),
         cvID = sample(round(nrow(fishnet) / 24), size=nrow(fishnet), replace = TRUE))

ggplot() +
  geom_sf(data = fire_net, aes(fill = countFires)) +
  geom_sf(data=main_streets, colour="white", size=1) +
  scale_fill_viridis(option = "magma",
                    name="Fire Count") +
  labs(title = "Count of fires for the fishnet",
       subtitle = "Market St. in white") +
  mapTheme()
```

## Download & wrangle risk factors

In this section, we download the independent variables. These are the predictive 'features' that measure geographic exposure to fires. Take note of the different strategies used to download the data. Some use `st_read` from the `sf` package, while others use `read.socrata`.

## Wrangle Land Use parcels

`landUse` parcels are downloaded first. Year built (`yrbuilt`) and building square footage (`bldgsqft`) is calculated, and the layer is then projected.

```r
landUse <- st_read("https://data.sfgov.org/api/geospatial/us3s-fp9q?method=export&format=GeoJSON") %>%
  filter(bldgsqft > 0) %>%
  filter(yrbuilt < 2022 & yrbuilt > 1900) %>%
  mutate(yrbuilt = as.numeric(yrbuilt), bldgsqft = as.numeric(bldgsqft)) %>%
  st_transform(st_crs(nhoods))
```

We might hypothesize that exposure to vacant parcels is a good predictor of fires. Vacant parcels are pulled out from `landUse` and converted to parcel centroids. The code block to follow then uses a spatial join to get the count of vacant parcels per grid cell. 

Note that we store all the features in the `vars_net` layer, which is an updated version of the `fishnet` layer. 

```r
vacant <- 
  filter(landUse, landuse == "VACANT") %>%
  st_centroid()

vars_net <-
  cbind(st_drop_geometry(fishnet),
    dplyr::select(vacant) %>%
    mutate(countVacants = 1) %>%
    aggregate(., fishnet, sum) %>%
    mutate(countVacants = replace_na(countVacants, 0))) %>%
  st_sf()
```

The code block below then maps the vacant buildings in San Francisco. 

What do you make of this measure of 'exposure'?

```r
ggplot() +
  geom_sf(data = vars_net, aes(fill = countVacants)) +
  geom_sf(data=main_streets, colour="white", size=1) +
  scale_fill_viridis(option = "magma",
                    name="Vacant Count") +
  labs(title = "Count of vacants for the fishnet",
       subtitle = "Market St. in white") +
  mapTheme()
```

The figure above shows that the count of vacant parcels may only describe exposure for grid cells that actually have vacants. Most grid cells return `0` vacant parcels. 

A better way to describe exposure is to measure the average distance from each `fishnet` grid cell _centroid_ to its _k_ nearest vacant neighbors. The code block below does this using the `nn_function` - a custom function created for this purpose. Here we set `k = 3`. Note, that the choice of _k_ affects the 'amount' of exposure.

We can now see that `vars_net` includes the count of vacant parcels as well as the nearest neighbor distance (`vacants.nn`) from every grid cell to its 3 nearest neighbors. `st_c` and `st_coid` is just shorthand for the two respective `sf` functions.

```r
st_c <- st_coordinates
st_coid <- st_centroid

vars_net <-
  vars_net %>%
    mutate(
      vacants.nn =
        nn_function(st_c(st_coid(vars_net)), st_c(vacant),3))

head(vars_net)
```

When the nearest neighbor distance is mapped, a measure of exposure can be seen for every location, Citywide. Why is this a better measure of exposure, compared to the count per grid cell?

```r
ggplot() +
  geom_sf(data = vars_net, aes(fill = vacants.nn)) +
  geom_sf(data=main_streets, colour="white", size=1) +
  scale_fill_viridis(option = "magma", direction = -1,
                    name="Vacant Distance") +
  labs(title = "Nearest neighbor distance to vacants",
       subtitle = "Market St.in white") +
  mapTheme()
```
Next, the mean year built and the mean building size is calculated for each grid cell by spatially joining the `landUse` parcel centroids to the `fishnet` using `aggregate`. Note the `mean` function.

```r
vars_net <-
  cbind(st_drop_geometry(vars_net),
    st_centroid(landUse) %>%
    dplyr::select(yrbuilt, bldgsqft) %>%
    aggregate(., fishnet, mean) %>%
    mutate(yrbuilt = replace_na(yrbuilt, 0),
           bldgsqft = replace_na(bldgsqft, 0))) %>%
  st_sf()
```

Parcels developed before 1900 are removed and the mean year is mapped. Here we hypothesize that building age is correlated with fire risk. 

```r
ggplot() +
    geom_sf(data=filter(vars_net, yrbuilt >= 1900), aes(fill = yrbuilt)) +
    geom_sf(data=main_streets, colour="white", size=1) +
    scale_fill_viridis(option = "magma",
                      name="Year") +
    labs(title = "Mean year built of parcels",
         subtitle = "Market St. in white") +
    mapTheme()
```

## Wrangle graffiti data

Graffiti is one measure of blight, and San Francisco collects geocoded grafitti reports in their 311 data. Here we grab 5000 random graffiti reports using the `limit` function in `read.socrata`. 

These points are mapped and they seem to cluster around the fire hotspots visualized above. This may not be the most thorough approach, but a random sample seems fitting, given how many graffiti reports have been filed.

```r
graf <- 
  read.socrata("https://data.sfgov.org/resource/vw6y-z8j6.json?$limit=5000&Category=Graffiti") %>%
  st_as_sf(coords = c("point.longitude", "point.latitude"), crs = 4326, agr = "constant") %>%
    st_transform(st_crs(nhoods)) %>%
    dplyr::select(requested_datetime) %>%
    .[nhoods,]

ggplot() +
  geom_sf(data = vars_net) +
  geom_sf(data=graf, size=.5) +
  labs(title = "Random sample of 311 graffiti points") +
  mapTheme()   
```
To create the graffiti features, the average nearest neighbor distance is measured using `k=5` and added to `vars_net`. `5` nearest neighbors is an arbitrary parameter, but it reflects the high relative density of graffiti. The analyst should experiment with other parameters of _k_.

```r
vars_net <-
  vars_net %>%
    mutate(
      graf.nn =
        nn_function(st_c(st_coid(vars_net)), st_c(graf),5))

ggplot() +
  geom_sf(data = vars_net, aes(fill = graf.nn)) +
  geom_sf(data=main_streets, colour="white", size=1) +
  scale_fill_viridis(option = "magma", direction = -1,
                    name="Vacant Distance") +
  labs(title = "Nearest neighbor distance to graffiti reports",
       subtitle = "Market St. in white") +
  mapTheme()
```

## Create the `final_net` 

Finally, we join the fires and risk factors into one `final_net`. By filtering `yrbuilt` to include grid cells with parcels developed after 1900 we omit areas without any building-oriented land uses (eg. Golden Gate Park).

```r
final_net <-
  left_join(fire_net, st_drop_geometry(vars_net), by="uniqueID") %>%
  filter(yrbuilt > 1900)
```

The code block below spatial joins `final_net` to the neighborhoods layer, giving a categorical neighborhood for each grid cell. This is required for the spatial cross validation performed below.

```r
final_net <-
  st_centroid(final_net) %>%
    st_join(nhoods) %>%
      st_drop_geometry() %>%
      left_join(dplyr::select(final_net, geometry, uniqueID)) %>%
      st_sf() %>%
  na.omit()

ggplot() +
  geom_sf(data = final_net, aes(fill = nhood), show.legend=FALSE) +
  geom_sf(data=main_streets, colour="white", size=1) +
  scale_fill_viridis(option = "magma", discrete = T,
                    name="Vacant Count") +
  labs(title = "Neighborhoods joined to the fishnet",
       subtitle = "Market St. in white") +
  mapTheme()
```

# 3. Exploring the spatial process of fires

The goal of this section is to engineer features for our model that help predict the fire hotspots and the fire coldspots. It is notoriously difficult for linear (regression) models to predict these areas, which are effectively outliers. It is important that we do not say, under-predict the hotspots - otherwise, we under-predict risk.

To do so, a statistic called Local Moran’s I posits a null hypothesis that fire count at a given location is randomly distributed relative to its immediate neighbors. Where this hypothesis can be overturned, we may observe local clustering - a hotspot.

The below analysis is also useful for exploratory purposes and will be familiar to many spatial analysts. What may be new, is using these metrics as features of a statistical model. The first step is to create a spatial weights matrix that relates each grid cell to its adjacent neighbors.

```r
final_net.nb <- poly2nb(as_Spatial(final_net), queen=TRUE)
final_net.weights <- nb2listw(final_net.nb, style="W", zero.policy=TRUE)
```

There is a lot in the code block below and more context can be found in [this section of the book](https://urbanspatial.github.io/PublicPolicyAnalytics/geospatial-risk-modeling-predictive-policing.html#exploring-the-spatial-process-of-burglary). The Local Moran's I statistic is calculated and a new feature is engineered called `Significant_Hotpots` - which are those statistically significant local clusters (_p <= 0.05_). 

The second part of the code block uses a loop to map four indicators each with their own unique legends. Not only does this analysis reveal the hotspots, but because `Significant_Hotpots` is observed for each grid cell, it can be used as a predictive feature. Let's see how below. 

```r
final_net.localMorans <- 
  cbind(
    as.data.frame(localmoran(final_net$countFires, final_net.weights)),
    as.data.frame(final_net)) %>% 
    st_sf() %>%
      dplyr::select(Fire_Count = countFires, 
                    Local_Morans_I = Ii, 
                    P_Value = `Pr(z > 0)`) %>%
      mutate(Significant_Hotspots = ifelse(P_Value <= 0.05, 1, 0)) %>%
      gather(Variable, Value, -geometry)
  
vars <- unique(final_net.localMorans$Variable)
varList <- list()

for(i in vars){
  varList[[i]] <- 
    ggplot() +
      geom_sf(data = filter(final_net.localMorans, Variable == i), 
              aes(fill = Value), colour=NA) +
      scale_fill_viridis(name="", option = "magma") +
      labs(title=i) +
      mapTheme() + theme(legend.position="bottom")}

do.call(grid.arrange,c(varList, ncol = 2, top = "Local Morans I statistics, Fires"))
```  
Measuring distance or exposure to these hotspots provides information that the predictive model can use to account for very local hotspots. Below, an average nearest neighbor feature is added to `final_net` describing distance to very `Significant_Hotpots` (note the p-value used).

```r
final_net <-
  final_net %>% 
  mutate(fires.isSig = 
           ifelse(localmoran(final_net$countFires, 
                             final_net.weights)[,5] <= 0.0000001, 1, 0)) %>%
  mutate(fires.isSig.dist = 
           nn_function(st_coordinates(st_centroid(final_net)),
                       st_coordinates(st_centroid(
                         filter(final_net, fires.isSig == 1))), 1))
```

We then map exposure to these very significant building fire hotspots.

```r
ggplot() +
  geom_sf(data = final_net, aes(fill = fires.isSig.dist), show.legend=FALSE) +
  geom_sf(data=main_streets, colour="white", size=1) +
  geom_sf(data = filter(final_net, fires.isSig == 1) %>% st_centroid()) +
  scale_fill_viridis(option = "magma", direction = -1,
                    name="Vacant Count") +
  labs(title = "Distance to very significant fire hotpots", 
       subtitle = "Black points represent significant hotspots") +
  mapTheme()
```

# 4. Modeling & Validation

In this section we introduce the concept of spatial cross-validation - a special flavor of cross-validation. The risk prediction model is then compared to the Kernel Density to test for its planning utility. 

When the dependent variable is count data, a class of statistical model called Generalized Linear Model or `glm` is often used. There are many flavors of `glm` models covered throughout the book, and very little context is given here. 

While our model will be judged in a planning context, many readers will appreciate the regression summary below. Again, we are regressing fire count as a function of the features engineered above. The output shows that many of features are statistically significant predictors of fire counts.

```r
summary(glm(countFires ~ ., family = "poisson", 
          data = final_net %>% st_drop_geometry() %>%
            dplyr::select(-uniqueID, -cvID, -nhood, -countVacants)))
```            

## Spatial cross-validation

We are estimating a regression model predicting the latent risk of fires. In the video accompanying this workshop, I suggest that one of our goals, and a big theme of the book, is that our model be generalizable. This means that it performs with equity across all neighborhoods regardless of factors like race, class or even the density of building fires. 

A robust predictive model generalizes the fire risk ‘experience’ at the city and (for each) neighborhood scale. The best way to test for this is to hold out one neighborhood, train the model on those remaining, predict for the hold out, and record the goodness of fit. This makes sure that when predicting for any one neighborhood, its prediction will be derived from the collective experience of all the other neighborhoods. 'Leave-One-Group-Out spatial cross-validation' (LOGO-CV) is this process and we will do so with each neighborhood taking a turn as a hold-out.

Imagine that one neighborhood has a particularly unique experience - like the Tenderloin, which has most of the building fires. When predicting for the Tenderloin, we actually ignore the Tenderloin experience by holding it out, relying on the other neighborhood experiences to predict for it. You can see how this is a very rigid assumption, but one that helps ensure generalizability.

We run the spatial cross-validation twice - once with the spatial process variables (eg. `fires.isSig`), and once without. The code block below creates these lists of features.

```r
reg.vars <- c("vacants.nn", "yrbuilt", "bldgsqft", "graf.nn")

reg.ss.vars <- c("vacants.nn", "yrbuilt", "bldgsqft", "graf.nn",
                 "fires.isSig", "fires.isSig.dist")
```

The code block below performs the spatial cross validation. It iteratively pulls out one neighborhood or `id`, trains the model on the remaining groups, and then predicts for the hold out. It outputs predictions for each grid cell.

```r
crossValidate.fire <- function(dataset, id, dependentVariable, indVariables) {
  
  allPredictions <- data.frame()
  cvID_list <- unique(dataset[[id]])
  
  for (i in cvID_list) {
    
    thisFold <- i
    cat("This hold out fold is", thisFold, "\n")
    
    fold.train <- filter(dataset, dataset[[id]] != thisFold) %>% as.data.frame() %>% 
      dplyr::select(id, geometry, indVariables, dependentVariable)
    fold.test  <- filter(dataset, dataset[[id]] == thisFold) %>% as.data.frame() %>% 
      dplyr::select(id, geometry, indVariables, dependentVariable)
    
    regression <-
      glm(countFires ~ ., family = "poisson", 
          data = fold.train %>% 
            dplyr::select(-geometry, -id))
    
    thisPrediction <- 
      mutate(fold.test, Prediction = predict(regression, fold.test, type = "response"))
    
    allPredictions <-
      rbind(allPredictions, thisPrediction)
    
  }
  return(st_sf(allPredictions))
} 
```

The code block below runs the cross-validation. The first chunk omits the spatial process variables, and the second includes it. In both cases the holdout group `id` is `nhood`, a field in `final_net`. 

```r
reg.spatialCV <- crossValidate.fire(
  dataset = final_net,
  id = "nhood",
  dependentVariable = "countFires",
  indVariables = reg.vars) %>%
    dplyr::select(cvID = nhood, countFires, Prediction, geometry)

reg.ss.spatialCV <- crossValidate.fire(
  dataset = final_net,
  id = "nhood",
  dependentVariable = "countFires",
  indVariables = reg.ss.vars) %>%
    dplyr::select(cvID = nhood, countFires, Prediction, geometry)
```

## Analyze model errors

Let us examine model accuracy by calculating model errors, subtracting grid cell `Prediction`s from the observed `countFires`. `reg.summary` includes `countFires`, predictions, errors and a `Regression` label for both regressions estimated above.  

```r
reg.summary <- 
  rbind(
    mutate(reg.spatialCV,    Error = Prediction - countFires,
                             Regression = "Spatial LOGO-CV: Just Risk Factors"),
    mutate(reg.ss.spatialCV, Error = Prediction - countFires,
                             Regression = "Spatial LOGO-CV: Spatial Process")) %>%
    st_sf() 

as.tibble(st_drop_geometry(head(reg.summary)))
```

The code block below calculates several goodness of fit metrics for both regressions and by `nhood` (the `cvID`). `Mean_Error` allows us to understand if we are over or under predicting in a given place. `MAE` takes the absolute value of errors. 

```r
error_by_reg_and_fold <- 
  reg.summary %>%
    group_by(Regression, cvID) %>% 
    summarize(Mean_Error = mean(Prediction - countFires, na.rm = T),
              MAE = mean(abs(Mean_Error), na.rm = T),
              SD_MAE = mean(abs(Mean_Error), na.rm = T)) %>%
  ungroup()
```

Neighborhood errors are now mapped for both regressions. What do you notice about their spatial distribution? In general, errors are much lower when the local spatial process is accounted for - ie. the hot and cold spots. Without those features, we get higher errors in and around the Tenderloin. 

```r
error_by_reg_and_fold %>%
  ggplot() +
    geom_sf(aes(fill = MAE)) +
    facet_wrap(~Regression) +
    scale_fill_viridis(option = "magma", direction = -1) +
    labs(title = "Burglary errors by LOGO-CV Regression") +
    mapTheme() + theme(legend.position="bottom")   
```

## Model predictions

Although this is a very simple model, assuming it is satisfactory, we can move on to the model predictions. The visualization below maps risk predictions, the Kernel Density and the observed `countFires`. 

The risk prediction model forecasts much higher risk in and around the Tenderloin, so much so that it masks colors associated with marginal risk in other areas of the City. The hotspots from the Kernel Density seem much more apparent. 

So which predicts _better_ - the Kernel Density or the Risk Prediction Model? Let's find out.

```r
grid.arrange(ncol=1,    
  reg.summary %>%
    group_by(Regression, cvID) %>%
    filter(str_detect(Regression, "Spatial Process")) %>%
    ggplot() + geom_sf(aes(fill=Prediction)) +
    scale_fill_viridis(option = "magma", direction = -1) +
    ggtitle("Risk Predictions") + mapTheme(),
  as.data.frame(fire_KD) %>%
    st_as_sf(coords = c("x", "y"), crs = st_crs(nhoods)) %>%
    aggregate(., final_net, mean) %>%
      ggplot() +
        geom_sf(aes(fill=value)) +
        scale_fill_viridis(name = "Density", option = "magma", direction = -1) +
        ggtitle("Kernel Density") + mapTheme(),
  reg.summary %>%
    group_by(Regression, cvID) %>%
    filter(str_detect(Regression, "Spatial Process")) %>%
    ggplot() + geom_sf(aes(fill=countFires)) +
    scale_fill_viridis(option = "magma", direction = -1) +
    ggtitle("Observed fire count") + mapTheme())
```     

# 5. Risk Prediction or Kernel Density?

As we have discussed, the most 'accurate' model may not the best. In fact, the most genearlizable model - which is less bias, might be important, but also may not be the best contender. What we are really after is the most 'useful' model - and useful can only be judged in the context of the planning use case. If we were on the ground in San Francisco, we might study how the Fire Department currently makes decisions around fire inspection and comparable use cases. 

In this case, we assume a Fire Analyst uses Kernel Density to understand areas at most risk for building fires. If density is the 'business-as-usual' approach for targeting resources, then our forecast must be more useful - meaning it must identify more locations as high risk that actually did experience fires. 

To make it more challenging, we test our 2017-2020 fire model on the 2021 fires withheld above. This way we can see if our model generalizes to the next year - which is important for the Fire Department's planning process.

Consider the Kernel Density and the risk predictions to be “predictive surfaces” - layers to describe risk Citywide. Below, each surface is divided into 5 risk categories, with the 90% to 100% risk category as the highest. `ntile` calculates percentiles. You can see, `aggregate` is used to spatial join the 2021 fires to the risk category grid cells. This is done first for the Kernel Density, below.

```r
fire_KDE_sf <- as.data.frame(fire_KD) %>%
  st_as_sf(coords = c("x", "y"), crs = st_crs(final_net)) %>%
  aggregate(., final_net, mean) %>%
  mutate(label = "Kernel Density",
         Risk_Category = ntile(value, 100),
         Risk_Category = case_when(
           Risk_Category >= 90 ~ "90% to 100%",
           Risk_Category >= 70 & Risk_Category <= 89 ~ "70% to 89%",
           Risk_Category >= 50 & Risk_Category <= 69 ~ "50% to 69%",
           Risk_Category >= 30 & Risk_Category <= 49 ~ "30% to 49%",
           Risk_Category >= 1 & Risk_Category <= 29 ~ "1% to 29%")) %>%
  cbind(
    aggregate(
      dplyr::select(fires21) %>% mutate(fireCount = 1), ., sum) %>%
    mutate(fireCount = replace_na(fireCount, 0))) %>%
  dplyr::select(label, Risk_Category, fireCount)
```

Now for the risk predictions, below.

```r
fire_risk_sf <-
  filter(reg.summary, Regression == "Spatial LOGO-CV: Spatial Process") %>%
  mutate(label = "Risk Predictions",
         Risk_Category = ntile(Prediction, 100),
         Risk_Category = case_when(
           Risk_Category >= 90 ~ "90% to 100%",
           Risk_Category >= 70 & Risk_Category <= 89 ~ "70% to 89%",
           Risk_Category >= 50 & Risk_Category <= 69 ~ "50% to 69%",
           Risk_Category >= 30 & Risk_Category <= 49 ~ "30% to 49%",
           Risk_Category >= 1 & Risk_Category <= 29 ~ "1% to 29%")) %>%
  cbind(
    aggregate(
      dplyr::select(fires21) %>% mutate(fireCount = 1), ., sum) %>%
      mutate(fireCount = replace_na(fireCount, 0))) %>%
  dplyr::select(label,Risk_Category, fireCount)  
```

Next, the 5 risk categories are mapped along with the 2021 fires overlaid in red. Which predictive surface looks like the better targeting tool? 

```r
rbind(fire_KDE_sf, fire_risk_sf) %>%
  na.omit() %>%
  gather(Variable, Value, -label, -Risk_Category, -geometry) %>%
  ggplot() +
    geom_sf(aes(fill = Risk_Category), colour = NA) +
    geom_sf(data = fires21, colour = "red") +
    facet_wrap(~label, ) +
    scale_fill_viridis(option = "magma", discrete = TRUE) +
    labs(title="Comparison of Kernel Density and Risk Predictions",
         subtitle="2017-2020 fire risk predictions; 2021 fires overlayed") +
    mapTheme()
```
The plot below takes the evaluation one step further, analyzing the rate of actual 2021 fires that fall into each risk category across both predictive surfaces. What can we conclude from the 5th and highest risk category and how might this speak to the utility of the algorithm?

```r
rbind(fire_KDE_sf, fire_risk_sf) %>%
  st_set_geometry(NULL) %>% na.omit() %>%
  gather(Variable, Value, -label, -Risk_Category) %>%
  group_by(label, Risk_Category) %>%
  summarize(countFires = sum(Value)) %>%
  ungroup() %>%
  group_by(label) %>%
  mutate(Rate_of_test_set_fires = countFires / sum(countFires)) %>%
    ggplot(aes(Risk_Category,Rate_of_test_set_fires)) +
      geom_bar(aes(fill=label), position="dodge", stat="identity") +
      #scale_fill_viridis(option="magma", discrete = TRUE, direction=-1) +
      scale_fill_manual(values = c("#6b4596ff", "#f7cb44ff")) + 
      labs(title = "Risk prediction vs. Kernel density, 2021 Fires") +
      plotTheme() + theme(axis.text.x = element_text(angle = 45, vjust = 0.5))
```
Indeed, the risk predictive model captures more 2021 fires than the Kernel Density, suggesting it is a more useful tool for planning. Recall, the Kernel Density predicts based purely on spatial autocorrelation. To reiterate from the introduction - the difference is that Kernel Density tells us to revisit past fires, but risk predictive models, if properly developed, reveal risk based on a more thorough predictive understanding of the spatial system and why things occur where.

In the video workshop that accompanies this markdown, I argue that risk prediction models are much more effective when we have a complete sample of the outcome - like building fires. The analyst should be cautious of use cases where the dependent variable suffers from selection bias - like drug crimes.

I hope this markdown/workshop was useful and I hope you replicate these methods in your own work. Please do check out my new book, [Public Policy Analytics](https://urbanspatial.github.io/PublicPolicyAnalytics/index.html), to learn so much more about both spatial and aspatial modeling approaches to solving public policy challenges. 

Thanks!