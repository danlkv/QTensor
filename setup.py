import setuptools

# Configure the required packages and scripts to install.
REQUIRED_PACKAGES = [
    'numpy',
    'networkx>=2.3',
    'matplotlib'
    ,'google-api-core[grpc]<=1.14.0'
    ,'cirq'
    ,'qiskit[optimization]'
    ,'pyrofiler>=0.1.5'
    ,'loguru'
    ,'tqdm'
    ,'click'
    ,'qtensor-qtree'
    ,'lazy-import'
    ,'pynauty'
    ,'docplex'
    ,'scipy'

]

setuptools.setup(name='qtensor',
                 version='0.1.2',
                 description='Framework for efficient quantum circuit simulations',
                 url='https://github.com/danlkv/qtensor',
                 keywords='quantum_circuit quantum_algorithms',
                 author='D. Lykov, et al.',
                 author_email='dan@qtensor.org',
                 license='Apache',
                 packages=setuptools.find_packages(),
                 install_requires=REQUIRED_PACKAGES,
                 extras_require={
                     'tensorflow': ['tensorflow<=1.15'],
                 },
                 include_package_data=True,
                 zip_safe=False)
