# MTS-ANO2
MTS-ANO2 is a project based on MTS-ANO

# Getting Started
git clone https://gitee.com/csqgitbelong/mts-ano2.git

conda env create -f env_yml/"environment.yml" -n "new_environment_name"

# To Develop

## Before
1. git checkout master
2. git branch -d "your last develop branch"
3. git pull origin master
4. git checkout -b "your new develop branch"

## After
1. git add .
2. git commit -m "中文提交信息"
3. git pull --rebase origin master
4. git push origin "your develop branch"

# File
1. Create "imgs" and "out" folder in the model folder to store the experiment results
2. Create "test" folder under in model folder to store test scripts
