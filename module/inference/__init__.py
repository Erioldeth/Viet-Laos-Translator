from module.inference.beam_search import BeamSearch
from module.inference.beam_search2 import BeamSearch2
from module.inference.decode_strategy import DecodeStrategy

strategies = {
	"BeamSearch": BeamSearch,
	"BeamSearch2": BeamSearch2
}
