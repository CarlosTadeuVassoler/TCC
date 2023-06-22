# CMakeList.txt: projeto de CMake para CarregandoModelo, incluir origem e definir
# lógica específica a um projeto aqui.
#

set(CMAKE_PREFIX_PATH "C:\\Users\\Carlos\\Desktop\\TCC\\libtorch\\libtorch")
find_package(Torch REQUIRED)

# Adicione a origem ao executável deste projeto.
add_executable (CarregandoModelo "CarregandoModelo.cpp" "CarregandoModelo.h")
target_link_libraries(CarregandoModelo "${TORCH_LIBRARIES}")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CarregandoModelo PROPERTY CXX_STANDARD 20)
endif()

if (MSVC)
	file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	add_custom_command(	TARGET CarregandoModelo
						POST_BUILD
						COMMAND ${CMAKE_COMMAND} -E copy_if_different
						${TORCH_DLLS}
						$<TARGET_FILE_DIR:CarregandoModelo>)
endif (MSVC)

# TODO: Adicione testes e instale destinos, se necessário.
