// nxbench custom JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // Add custom functionality here
    console.log('nxbench documentation loaded');

    // Example: Add a click event to all buttons with class 'nxbench-button'
    const buttons = document.querySelectorAll('.nxbench-button');
    buttons.forEach(button => {
        button.addEventListener('click', function (event) {
            event.preventDefault();
            console.log('nxbench button clicked:', this.href);
            window.open(this.href, '_blank');
        });
    });
});
