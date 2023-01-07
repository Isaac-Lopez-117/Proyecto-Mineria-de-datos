const btnsEliminacion = document.querySelectorAll(".Eliminar");

(function () {
  btnsEliminacion.forEach((btn) => {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      Swal.fire({
        title: "¿Esta seguro de eliminar el proyecto?",
        showCancelButton: true,
        confirmButtonText: "Eliminar",
        confirmButtonColor: "#d33",
        backdrop: true,
        showLoaderOnConfirm: true,
        preConfirm: () => {
          location.href = e.target.href;
        },
        allowOutsideClick: () => false,
        allowEscapeKey: () => false,
      });
    });
  });
})();
