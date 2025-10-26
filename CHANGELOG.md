# Changelog

All notable changes to this project will be documented in this file.

## [0.7.3] - 2025-10-25

### Updated
- **VertexAI Provider**: Latest improvements and refinements
  - Enhanced tool call handling and response processing
  - Optimized Content object creation
  - Improved error handling and logging
  
### Maintenance
- Code quality improvements
- Updated dependencies
- Build verification and testing

## [0.7.2] - 2025-10-25

### Fixed
- **VertexAI Gemini Tools**: Complete tool call support implementation
  - Fixed function_call parsing from response.candidates[0].content.parts
  - Proper tool_calls extraction with id, name, and arguments
  - Tool calls now correctly returned in GenerationResponse
  - System instruction properly passed to generate_content API

- **All Previous v0.7.1 Fixes Included**:
  - VertexAI "contents must not be empty" error resolved
  - OpenAI role enum/string hybrid support
  - Tool runner assistant message inclusion
  - Streaming tool call support

### Package
- Rebuilt with all latest changes
- Clean build verification
- PyPI upload with complete implementation

## [0.7.1] - 2025-10-24

### Fixed
- **VertexAI Gemini Tools**: Fixed critical "contents must not be empty" error
  - Proper Content object creation with `genai_types.Content` and `genai_types.Part`
  - Tool messages (TOOL role) now correctly mapped to Gemini format
  - System instruction moved to proper `system_instruction` parameter
  - Tool call parsing from response working correctly with `part.function_call`

- **OpenAI Provider**: Fixed role handling for mixed enum/string values
  - Added `hasattr(msg.role, 'value')` check for hybrid enum/string support
  - Backward compatible with both MessageRole enum and string role values

- **Tool Runner Utility**: Fixed missing assistant message in tool execution loop
  - Assistant's message with tool calls now properly added to history
  - Tool results correctly appended after assistant message
  - Fixes conversation continuity in multi-turn tool interactions

### Improved
- **VertexAI Streaming**: Full streaming support with tool calls
  - Tool calls detected during streaming
  - Proper StreamChunk generation with tool_call parameter
  - Usage metadata correctly propagated in final chunk

- **Documentation**: Added comprehensive VertexAI tools usage examples in README

### Technical Details
- All 3 critical bugs fixed in v0.7.1
- Lint checks passing (flake8, black)
- Build verified and tested

## [0.4.1] - 2025-10-09

### Fixed
- **Docker Compatibility**: Fixed pydantic dependency conflict for Docker deployments
  - Updated `anthropic` dependency from `>=0.3.0` to `>=0.39.0` (pydantic v2 compatible)
  - Ensures compatibility with modern deployment environments
  - Resolves dependency conflicts in containerized environments

### Dependencies
- anthropic: `>=0.3.0` → `>=0.39.0` (pydantic v2 support)

### Docker Deployment
This version is specifically optimized for Docker and containerized deployments.

## [0.4.0] - 2025-10-08

### Added
- **Ollama Provider**: Complete local LLM support
  - Support for 16+ Ollama models (llama3.1, llama2, codellama, etc.)
  - Full streaming support with real-time chunks
  - Conversation and basic generation capabilities
  - HTTP API integration at localhost:11434

### Features
- Unified interface for both cloud and local LLMs
- Factory pattern for easy provider switching
- Comprehensive error handling
- Type safety with Pydantic v2

### Providers
- ✅ OpenAI (GPT-3.5, GPT-4, GPT-4o)
- ✅ Anthropic (Claude-3 models)
- ✅ Google Gemini
- ✅ VertexAI (Mistral models)
- ✅ Ollama (Local LLMs)

### Tested
- All 5 providers working ✅
- Streaming support ✅
- Local LLM integration ✅
- Comprehensive test suite ✅

## [0.3.3] - 2025-10-07

### Added
- VertexAI provider with Mistral support
- Complete async/await implementation
- Streaming capabilities

### Fixed
- Abstract method implementations
- Provider initialization issues

## [0.3.0] - 2025-10-06

### Added
- Initial release with OpenAI, Anthropic, and Gemini providers
- Factory pattern implementation
- Comprehensive configuration system