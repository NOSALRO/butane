#pragma once

#include <algorithm>
#include <boost/optional.hpp>
#include <cassert>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include "utils.hpp"
#include "anymodulelist.hpp"
#include "autoencoder.hpp"
#include "conv_block.hpp"
#include "mlp_block.hpp"
#include "quantizer.hpp"
#include "trainer.hpp"
