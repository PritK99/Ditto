#include <clang/Tooling/Tooling.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/ASTConsumer.h>

#include <iostream>
#include <fstream>
#include <sstream>

class MyVisitor
    : public clang::RecursiveASTVisitor<MyVisitor> {

public:

    clang::ASTContext* Context;

    explicit MyVisitor(clang::ASTContext* Context)
        : Context(Context) {}

    bool VisitFunctionDecl(clang::FunctionDecl* f) {

        auto& SM = Context->getSourceManager();

        if (!SM.isWrittenInMainFile(f->getLocation()))
            return true;

        std::cout
            << "FunctionDecl: "
            << f->getNameAsString()
            << "\n";

        return true;
    }

    bool VisitVarDecl(clang::VarDecl* v) {

        auto& SM = Context->getSourceManager();

        if (!SM.isWrittenInMainFile(v->getLocation()))
            return true;

        std::cout
            << "VarDecl: "
            << v->getNameAsString()
            << "\n";

        return true;
    }

    bool VisitStringLiteral(clang::StringLiteral* s) {

        auto& SM = Context->getSourceManager();

        if (!SM.isWrittenInMainFile(s->getBeginLoc()))
            return true;

        std::cout
            << "StringLiteral: "
            << s->getString().str()
            << "\n";

        return true;
    }

    bool VisitCXXOperatorCallExpr(
        clang::CXXOperatorCallExpr* op) {

        auto& SM = Context->getSourceManager();

        if (!SM.isWrittenInMainFile(op->getExprLoc()))
            return true;

        std::cout
            << "OperatorCall: "
            << getOperatorSpelling(op->getOperator())
            << "\n";

        return true;
    }
};

class MyConsumer : public clang::ASTConsumer {

public:

    void HandleTranslationUnit(
        clang::ASTContext& Context) override {

        MyVisitor visitor(&Context);

        visitor.TraverseDecl(
            Context.getTranslationUnitDecl());
    }
};

class MyFrontendAction
    : public clang::ASTFrontendAction {

public:

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(
        clang::CompilerInstance& CI,
        llvm::StringRef file) override {

        return std::make_unique<MyConsumer>();
    }
};

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: parser <file.cpp>\n";
        return 1;
    }

    std::ifstream in(argv[1]);

    std::stringstream buffer;
    buffer << in.rdbuf();

    std::string code = buffer.str();

    clang::tooling::runToolOnCode(
        std::make_unique<MyFrontendAction>(),
        code);

    return 0;
}