console.log("============================================================");
console.log("Constraining the type of T");
console.log("============================================================");
// This class takes an array of strings or numbers and concatenates them into a single string
var Concatenator = /** @class */ (function () {
    function Concatenator() {
    }
    // Method that concatenates the array of items into a string
    Concatenator.prototype.concatenateArray = function (items) {
        // Initialize an empty string to store the concatenated values
        var returnString = "";
        // Loop through each item in the array
        for (var i = 0; i < items.length; i++) {
            // If this is not the first item, add a comma before appending the value
            returnString += i > 0 ? "," : "";
            // Append the current value to the return string
            returnString += items[i].toString();
        }
        // Return the final concatenated string
        return returnString;
    };
    return Concatenator;
}());
// Usage example
// Create a new instance of the Concatenator class
var concator = new Concatenator();
// Concatenate an array of strings
var concatResult = concator.concatenateArray(["first", "second", "third"]);
console.log("concatResult = ".concat(concatResult));
// Concatenate an array of numbers
concatResult = concator.concatenateArray([1000, 2000, 3000]);
console.log("concatResult = ".concat(concatResult));
// Concatenate an array of numbers
concatResult = concator.concatenateArray(["1000", 2000, 3000]);
console.log("concatResult = ".concat(concatResult));
// Invalid assignment
// Type 'boolean' is not assignable to type 'string | number'
// concatResult = concator.concatenateArray([
//     true, false, true
// ]);
