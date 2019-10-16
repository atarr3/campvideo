from setuptools import setup

setup(
    name='campvideo',
    version='0.3.0',
    description='Analyzes political campaign advertisements.',
    long_description=open('README.rst').read() + '\n\n' + open('HISTORY.rst').read(),
    author='Alex Tarr',
    author_email='atarr3@gmail.com',
    url='https://github.com/atarr3/campvideo',
    packages=['campvideo'],
	package_data={'campvideo' : ['data/*.pb']},
    entry_points={
        'console_scripts': [
            'audio_feats=campvideo.audio_feats:main',
			'imtext=campvideo.imtext:main',
            'match_vids=campvideo.match_vids:main',
            'summarize_vids=campvideo.summarize_vids:main'
        ]
    },
    license='MIT License',
    install_requires=[
		"ffmpeg-python",
		"numpy",
		"opencv-python >= 3.4.7.28",
		"pandas >= 0.24.2",
		"scikit-learn >= 0.20.1",
		"scipy >= 1.1.0"
    ]
)
