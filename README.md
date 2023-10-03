# Q-Switch Mixture of Policies Hindsight Experience Replay (QMP-HER)

It is based on the implementation of the G-HGG paper: https://github.com/hk-zh/C-HGG

## Requirements
1. Ubuntu 16.04 or macOS Catalina 10.15.7 (newer versions also work well) 
2. Python 3.5.2 (newer versions such as 3.6.8 should work as well, 3.8 or higher is not suggested)
3. MuJoCo == 2.00 (see instructions on https://github.com/openai/mujoco-py)
4. Install gym from https://github.com/franroldans/custom_gym. Certain environment specifications and parameters are set there. 

## Training under different environments

### Fetch Environments
```bash
# FetchPickAndPlace
python train.py --tag pickplace --goal vanilla --learn maxq --env FetchPickAndPlace-v1  --batch_size 64 --buffer_size 500 --epoch 20  --reach_primitive_path <PATH_TO_REACH_PRIMITIVES_LOG_DIR>
# FetchPush
python train.py --tag push --goal vanilla --learn maxq --env FetchPush-v1  --batch_size 64 --buffer_size 500 --epoch 20 --reach_primitive_path <PATH_TO_REACH_PRIMITIVES_LOG_DIR>
# FetchPickObstacle
python train.py --tag maxqobstacle --learn pickmaxq --env FetchPickObstacle-v1 --goal custom --reach_primitive_path<PATH_TO_PICK_AND_PLACE_PRIMITIVES_LOG_DIR>
# FetchPickAndThrow
python train.py --tag pickthrow --learn pickmaxq --env FetchPickAndThrow-v1 --goal custom --reach_primitive_path <PATH_TO_PICK_AND_PLACE_PRIMITIVES_LOG_DIR>  --push_primitive_path <PATH_TO_PICK_OBSTACLE_PRIMITIVES_LOG_DIR>

```

