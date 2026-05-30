console.log("============================================================");
console.log("Using a function with generic syntax");
console.log("============================================================");
// Define a function named "printGeneric" that takes in a generic type "T" as a parameter
function printGeneric(value) {
    // Log the type of "T" and the value of "value" to the console
    console.log("typeof T is : ".concat(typeof value));
    console.log("value is : ".concat(value));
}
// Call the printGeneric function with various argument types
printGeneric(1);
printGeneric("test");
printGeneric(true);
printGeneric(function () { });
printGeneric({ id: 1 });
console.log("============================================================");
console.log("Explicitly specifying the type in generic syntax");
console.log("============================================================");
// Explicitly specifying the type in generic syntax
printGeneric("test");
// printGeneric<number>("100")  // error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.
