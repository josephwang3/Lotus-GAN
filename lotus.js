// Joseph Wang
// 12/27/2021
// Short JavaScript script to change image on "This Lotus Does Not Exist" with every reload

// function to generate a random integer
function generateRandomInteger(min, max) {
    return Math.floor(min + Math.random()*(max + 1 - min));
}

// build image path
ref = "results/image" + generateRandomInteger(0, 133) + ".png";
document.getElementById("lotusimg").src = ref;