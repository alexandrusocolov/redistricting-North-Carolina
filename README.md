# Developing Weighted K-Means algorithm for Redistricting North Carolina

Developing an unsupervised ML method for drawing the voting districts in North Carolina. 

![](images/map.png)

**Algorithm**

The original Weighted K-Means algorithm has been proposed by Guest et al. (2019) and the original paper can be found here: https://link.springer.com/article/10.1007/s42001-019-00053-9

I extend the method by weighting the distances not only by population but also by racial and partisan differences. Moreover, I open-source my code in hopes to continue the conversation about computational redistricting. 


**Data**

 `centroids.csv`: Longitude and latitude for each Voting Tabulation District (VTD), the most granular data I could get
 
 `demo_NC_VTD.cpg`, `demo_NC_VTD.dbf`, `demo_NC_VTD.prj`, `demo_NC_VTD.shp` and `demo_NC_VTD.shx`: VTD level population, racial and voting data as well as shapes of each VTD for plotting. `centroids.csv` is a derivative of these shapes, so the order of rows is the same. 
 
`demo_SSP_asrc.csv`: Projected zip-code level population data from https://osf.io/uh5sj/ subsetted for North Carolina in 2030 and 2050

**Demonstration**
`weighted_kmeans.py` is a Python script that implements the Weighted K-means from scratch.

A demonstration of how to use the method can be found in `demo_weighted_kmeans.ipynb`. The notebook also includes visualizations of the proposed maps for 2016 as well as the projected maps in 2030 and 2050 based on this paper https://www.nature.com/articles/sdata20195. 
