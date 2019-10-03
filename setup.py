from setuptools import setup

setup(
    name='campvideo',
    version='0.1.0',
    description='Analyzes political campaign advertisements.',
    long_description=open('README.rst').read(),
    author='Alex Tarr',
    author_email='atarr3@gmail.com',
    url='https://github.com/atarr3/campvideo',
    packages=['campvideo'],
    scripts=['bin/audio_feats.py',
		     'bin/match_vids.py',
			 'bin/summarize_vids.py'],
    license='MIT License',
    install_requires=[
		"ffmpeg-python",
		"numpy",
		"opencv-python >= 4.1.1.26",
		"pandas >= 0.25.1",
		"scikit-learn >= 0.20.1",
		"scipy >= 1.1.0"
    ]
)
