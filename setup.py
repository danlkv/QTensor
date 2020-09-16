import setuptools

# Configure the required packages and scripts to install.
REQUIRED_PACKAGES = [
    'numpy>=1.18.1',
    'networkx>=2.3',
    'matplotlib>=3.1.3'
    ,'google-api-core[grpc]<=1.14.0'
    ,'cirq'
    ,'qiskit==0.17.0'
    ,'pyrofiler>=0.1.5'
    ,'loguru'
    ,'tqdm'

]

setuptools.setup(name='qensor',
                 version='0.1.1',
                 description='Framework for efficient quantum circuit simulations',
                 url='https://github.com/DaniloZZZ/qensor',
                 keywords='quantum_circuit quantum_algorithms',
                 author='D. Lykov, et al.',
                 author_email='dan@qensor.org',
                 license='Apache',
                 packages=setuptools.find_packages(),
                 install_requires=REQUIRED_PACKAGES,
                 extras_require={
                     'tensorflow': ['tensorflow<=1.15'],
                 },
                 include_package_data=True,
                 zip_safe=False)
