import time
from gmacc import config
from gmacc.dbsynth import synthetic_database


args = config.SyntheticDatabase(config_path='synthetic_database_config.yaml').get_config()
print(args)
absRefTime = time.time()
synthetic_database.create(args)
print('Script running time:', time.time() - absRefTime)
