console.log("============================================================");
console.log("Construct a generic type from another");
console.log("============================================================");
// Define a function named printProperty that takes two generic type parameters
function printProperty(object, key) {
    var propertyValue = object[key];
    // @ts-ignore
    console.log("object[".concat(key, "] = ").concat(propertyValue));
}
// tsconfig required
// {
//     "compilerOptions": {
//       "keyofStringsOnly": true
//     }
// }
// Define an object with a unique id, name and print method
var obj1 = {
    id: 1,
    name: "myName",
    print: function () {
        console.log("".concat(this.id));
    },
};
// Call the function to print a property of an object
printProperty(obj1, "id");
printProperty(obj1, "name");
// error TS2345: Argument of type '"name1"' is not assignable to parameter of type '"name" | "id" | "print"'.
// printProperty(obj1, "name1");
console.log("============================================================");
console.log("what happens when we use this function to print the 'print' method property");
console.log("============================================================");
printProperty(obj1, "print");
