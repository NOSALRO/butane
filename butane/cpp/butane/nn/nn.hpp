#pragma once

// clang-format off
#include <algorithm>
#include <boost/optional.hpp>
#include <cassert>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include "utils/utils.hpp"
#include "anymodulelist.hpp"
#include "modules/modules.hpp"
#include "modules/conv_block.hpp"
#include "modules/mlp_block.hpp"
#include "modules/quantizer.hpp"
#include "architectures/autoencoders.hpp"
#include "utils/model_wrapper.hpp"
#include "utils/trainer.hpp"
// clang-format on
