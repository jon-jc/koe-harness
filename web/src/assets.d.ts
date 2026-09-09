/**
 * Non-code imports.
 *
 * esbuild resolves a `.css` import to a side effect that lands in the output
 * stylesheet; TypeScript needs telling that such a module exists at all. The
 * shape is deliberately empty — nothing should read a value from one.
 */
declare module "*.css" {
  const styles: void;
  export default styles;
}
