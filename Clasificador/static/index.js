document.addEventListener("DOMContentLoaded", function() {
    const botonesVerMas = document.querySelectorAll('.btn-ver-mas');

    botonesVerMas.forEach(function(boton) {
        

        boton.addEventListener('click', function(evento) {
            
            evento.preventDefault();
           
            const tarjeta = evento.target.closest('.producto-card');
            
            tarjeta.classList.toggle('expandida');
            if (tarjeta.classList.contains('expandida')) {
               
                boton.textContent = 'Ver menos';
            } else {
               
                boton.textContent = 'Ver más';
            }
        });
    });


});
