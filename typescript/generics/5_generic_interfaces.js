console.log("============================================================");
console.log("Generic interfaces");
console.log("============================================================");
var LogClass = /** @class */ (function () {
    function LogClass() {
    }
    LogClass.prototype.logToConsole = function (iPrintObj) {
        iPrintObj.print();
    };
    return LogClass;
}());
// Usage example
var printObject = {
    print: function () {
        console.log("printObject.print() called");
    },
};
var logClass = new LogClass();
logClass.logToConsole(printObject);
console.log("============================================================");
console.log("Creating new objects within generics");
console.log("============================================================");
var ClassA = /** @class */ (function () {
    function ClassA() {
    }
    return ClassA;
}());
var ClassB = /** @class */ (function () {
    function ClassB() {
    }
    return ClassB;
}());
// function createClassInstance<T>
//  (arg1: T): T {
// /**
//  * 
//  *  error TS2351: This expression is not constructable.
//     Type '{}' has no construct signatures.
//  * 
//  */
// //  return new arg1(); // error : see below
// }
var classAInstance = createClassInstance(ClassA);
function createClassInstance(arg1) {
    return new arg1();
}
