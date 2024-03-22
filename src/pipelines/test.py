from data.loading import load_configuration, load_parquet

path, filename, config_settings = load_configuration()
data = load_parquet(path["interim"], filename["combined"], optimize_memory=True)
features = load_parquet(path["features"], filename["features"], optimize_memory=True)

print(data.head())
print(features.head())