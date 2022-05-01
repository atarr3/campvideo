from setuptools import setup

setup(
    name='campvideo',
    version='1.1.0',
    description='Analyzes political campaign advertisements.',
    long_description=open('README.rst').read() + '\n\n' + open('HISTORY.rst').read(),
    author='Alex Tarr',
    author_email='atarr3@gmail.com',
    url='https://github.com/atarr3/campvideo',
    packages=['campvideo'],
	package_data={
		'campvideo': ['data/*.csv', 'data/*.npy', 'models/*.pkl', 'models/*.joblib']
	},
    entry_points={
        'console_scripts': [
            'audio_feats=campvideo.audio_feats:main',
			'download_models=campvideo.download_models:main',
			'imtext=campvideo.imtext:main',
            'match_vids=campvideo.match_vids:main',
            'summarize_vids=campvideo.summarize_vids:main',
			'transcribe_vids=campvideo.transcribe_vids:main'
        ]
    },
    license='MIT License',
	python_requires='>= 3.9',
    install_requires=[
		"face_recognition",
		"ffmpeg-python",
		"google-cloud-videointelligence>=2.7",
		"google-cloud-vision>=2.7",
        "matplotlib",
        "seaborn>=0.11.0",
		"numpy>=1.21",
		"opencv-python>=4.5.5",
		"pandas>=1.4",
		"scikit-learn==1.0.1",
		"scipy>=1.7",
        "spacy==3.3.0"
    ]
)
