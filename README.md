# Linguistic Characterization of Populism in U.S. Presidential Campaigns

This repository contains the data and code underlying the doctoral thesis:

> Zanotto, Sergio Eugenio. *Linguistic Characterization of Populism in U.S. Presidential Campaigns*.
> Doctoral thesis (Dr. phil.), Faculty of Humanities, Department of Linguistics,
> University of Konstanz, 2026.

**Supervisors:** Miriam Butt (University of Konstanz) · Diego Frassinelli (LMU Munich) · David I. Beaver (UT Austin)

---

## Repository Structure

    python/
    ├── calculate_features/                       # Scripts for computing linguistic features from speech transcripts
    └── display_features/                         # Scripts for visualizing and exploring the extracted features
    ofOfficial_ideologyvspopulism.Rmd             # R code for final regression analyses
    Official_df_ideologyvspopulism.zip            # Final dataset for replication (use this one)
    main_data.zip                                 # Raw dataset with initial feature calculations

## Data

- **`Official_df_ideologyvspopulism.zip`** — Final, cleaned dataset containing all computed linguistic features and ideology scores. This is the file to use for replicating the regression analyses.
- **`main_data.zip`** — Raw dataset used as the starting point of the feature extraction pipeline. Not intended for direct use in statistical models.

## Pipeline

1. **Feature extraction** (`python/calculate_features/`) — Computes linguistic features from raw U.S. presidential campaign speech transcripts.
2. **Feature exploration** (`python/display_features/`) — Descriptive inspection and visualization of the extracted features.
3. **Regression analysis** (`ofOfficial_ideologyvspopulism.Rmd`) — Final statistical models estimating the relationship between linguistic features and populist ideology, run on `Official_df_ideologyvspopulism.zip`.

## Requirements

### Python
- Python 3.x

### R
- R 4.x
- Required packages listed at the top of `ofOfficial_ideologyvspopulism.Rmd`

## Citation

If you use this code or data, please cite:

> Zanotto, S. E. (2026). *Linguistic Characterization of Populism in U.S. Presidential Campaigns*. Doctoral thesis, University of Konstanz.

## Contact

## Contact

**Sergio Eugenio Zanotto**
University of Konstanz — Centre of Excellence "The Politics of Inequality"
Department of Linguistics / Department of Political Science
sergio.zanotto@uni-konstanz.de
