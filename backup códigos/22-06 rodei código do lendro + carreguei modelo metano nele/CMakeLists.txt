# CMakeList.txt: projeto de CMake para Heat-Int-IA, incluir origem e definir
# lógica específica a um projeto aqui.
#

set(CMAKE_PREFIX_PATH "C:\\Users\\Carlos\\Desktop\\TCC_bruto\\libtorch\\libtorch")
find_package(Torch REQUIRED)

# Adicione a origem ao executável deste projeto.
add_executable (Heat-Int-IA "Heat-Int-IA.cpp" "Heat-Int-IA.h")
target_link_libraries(Heat-Int-IA "${TORCH_LIBRARIES}")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Heat-Int-IA PROPERTY CXX_STANDARD 20)
endif()

if (MSVC)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	add_custom_command(	TARGET Heat-Int-IA
						POST_BUILD
						COMMAND ${CMAKE_COMMAND} -E copy_if_different
						${TORCH_DLLS}
						$<TARGET_FILE_DIR:Heat-Int-IA>)
endif (MSVC)

# TODO: Adicione testes e instale destinos, se necessário.
