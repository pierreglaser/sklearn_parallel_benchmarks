import os
if os.environ.get('ASV'):
    os.environ['SKLEARN_SITE_JOBLIB'] = os.path.join(os.environ['ASV_ENV_DIR'],
                                                     'project')
