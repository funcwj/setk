// test-complex.cc
// wujian@2018

#include "include/complex-base.h"
#include "include/beamformer.h"
#include "include/complex-vector.h"
#include "include/complex-matrix.h"

using namespace kaldi;

void test_cvector_init() {
    CVector<BaseFloat> v(10);
    v.SetRandn();
    std::cout << v;
    v.Resize(20, kCopyData);
    std::cout << v;
    v.Resize(8, kCopyData);
    std::cout << v;
    std::cout << v.Sum() << std::endl;
    v.Resize(10);
    std::cout << v;
    v(3, kReal) = 66.66;
    v(5, kImag) = 23.33;
    std::cout << v;
    Vector<BaseFloat> rv(10);
    rv.SetRandn();
    std::cout << rv;
    v.CopyFromVec(rv, kReal);
    std::cout << v;
    v.CopyFromVec(rv, kImag);
    std::cout << v;
    CVector<BaseFloat> t(rv);
    std::cout << t;

}

void test_cvector_addvec() {
    CVector<BaseFloat> v1(10), v2(10);
    for (int32 i = 0; i < 5; i++) {
        v1.SetRandn(), v2.SetRandn();
        std::cout << v1;
        std::cout << v2;
        v1.AddVec(1.0, 2.0, v2);
        std::cout << v1;
    }
}

void test_cvector_scale() {
    BaseFloat a, b;
    for (int32 i = 0; i < 5; i++) {
        int32 dim = Rand() % 6 + 2;
        CVector<BaseFloat> v(dim);
        v.SetRandn();
        std::cout << v;
        a = RandGauss(), b = RandGauss();
        std::cout << "(" << a << ", " << b << ")" << std::endl;
        v.Scale(a, b);
        std::cout << v;
    }
}

void test_cmatrix_scale() {
    BaseFloat a, b;
    for (int32 i = 0; i < 5; i++) {
        int32 r = Rand() % 6 + 2, c = Rand() % 6 + 2;
        CMatrix<BaseFloat> m(r, c);
        m.SetRandn();
        std::cout << m;
        a = RandGauss(), b = RandGauss();
        std::cout << "(" << a << ", " << b << ")" << std::endl;
        m.Scale(a, 0);
        std::cout << m;
    }
}



void test_cvector_mulelements() {
    CVector<BaseFloat> v1(6), v2(6);
    for(int32 i = 0; i < 10; i++) {
        v1.SetRandn(), v2.SetRandn();
        std::cout << v1;
        std::cout << v2;
        v1.MulElements(v2);
        std::cout << v1;
    }
}

void test_cvector_copy() {
    CVector<BaseFloat> v(8);
    for (int32 i = 0; i < 10; i++) {
        {
            CVector<BaseFloat> c(8);
            c.SetRandn();
            v = c;
            std::cout << c;
        }
        std::cout << v;
    }
}

void test_subcvector() {
    float buffer[4] = {1, 2, 3, 4};
    SubCVector<BaseFloat> buf(buffer, 2);
    std::cout << buf;
    for (int32 i = 0; i < 10; i++) {
        CVector<BaseFloat> v(10);
        v.SetRandn();
        SubCVector<BaseFloat> s(v, 2, 3);
        std::cout << v;
        std::cout << v.Range(2, 3);
        std::cout << s;
    }
}

void test_cvector_vecvec() {
    BaseFloat dr, di;
    for(int32 i = 0; i < 10; i++) {
        int32 dim = Rand() % 6 + 2;
        CVector<BaseFloat> v1(dim), v2(dim);
        v1.SetRandn(), v2.SetRandn();
        std::cout << "v1:\n" << v1;
        std::cout << "v2:\n" << v2;
        std::cout << "v1 * v2^" << (i % 2 ? "H": "T") << " = " << VecVec(v1, v2, (i % 2 ? kConj: kNoConj)) << std::endl;
    }
} 


void test_cvector_addmatvec() {
    for (int32 i = 0; i < 10; i++) {
        int32 r = Rand() % 6 + 2, c = Rand() % 6 + 2;
        CMatrix<BaseFloat> m1(r, c), m2(c, r);
        CVector<BaseFloat> v1(c), v2(r);
        v1.SetRandn(), v2.SetRandn();
        m1.SetRandn(), m2.SetRandn();
        std::cout << v1;
        std::cout << v2;
        std::cout << m1;
        std::cout << m2;
        v2.AddMatVec(1, 0, m1, kNoTrans, v1, 1, 1);
        std::cout << v2;
        v2.AddMatVec(1, 0, m2, kTrans, v1, 1, 1);
        std::cout << v2;
    }
}

void test_cmatrix_basic_op() {
    MatrixIndexT c = (Rand() % 8 + 1), r = (Rand() % 8 + 1);
    CMatrix<BaseFloat> cm(c, r);
    std::cout << cm.Info();
    std::cout << cm;
    cm.SetRandn();
    std::cout << cm;
    cm.Resize(c + 2, r + 1, kCopyData);
    std::cout << cm;
    std::cout << cm.Info();
    cm.Resize(c / 2, r / 2, kCopyData);
    std::cout << cm;
    std::cout << cm.Info();
    CMatrix<BaseFloat> sm(6, 5);
    sm.SetRandn();
    std::cout << sm;
    std::cout << "row 2:" << std::endl;
    std::cout << sm.Row(2);
    std::cout << "row range 1-3:" << std::endl;
    std::cout << sm.RowRange(1, 3);
    std::cout << "col range 2-4:" << std::endl;
    std::cout << sm.ColRange(2, 3);
    std::cout << "range (1, 2) - (4, 3):" << std::endl;
    std::cout << sm.Range(1, 4, 2, 2);
    sm.SetUnit();
    std::cout << sm;
    sm.SetRandn();
    std::cout << sm;
    CMatrix<BaseFloat> no_tr(sm, kNoTrans);
    std::cout << no_tr;
    CMatrix<BaseFloat> no_tr_cj(sm, kNoTrans, kConj);
    std::cout << no_tr_cj;
    CMatrix<BaseFloat> tr(sm, kTrans);
    std::cout << tr;
    CMatrix<BaseFloat> cj_tr(sm, kTrans, kConj);
    std::cout << cj_tr;
    Matrix<BaseFloat> m(5, 5);
    m.SetRandn();
    CMatrix<BaseFloat> cr(m);
    std::cout << m;
    std::cout << cr;
}

void test_cmatrix_addmatmat() {
    for (int32 i = 0; i < 10; i++) {
        int32 m = Rand() % 5 + 2, n = Rand() % 5 + 2, k = Rand() % 5 + 2;
        CMatrix<BaseFloat> cm1(m, n), cm2(n, k), ans(m, k);
        cm1.SetRandn(), cm2.SetRandn(), ans.SetRandn();
        std::cout << "ans:\n" << ans;
        ans.AddMatMat(2, 1, cm1, kNoTrans, cm2, kNoTrans, 0.5, 1);
        std::cout << "cm1:\n" << cm1;
        std::cout << "cm2:\n" << cm2;
        std::cout << "ans = (0.5+1i) * ans + (2+1i) * cm1 * cm2:\n" << ans;
    }    
}

void test_cmatrix_addvecvec() {
    for (int32 i = 0; i < 10; i++) {
        int32 r = Rand() % 5 + 2, c = Rand() % 5 + 2;
        CMatrix<BaseFloat> cm(r, c);
        cm.SetRandn();
        std::cout << "cm:\n" << cm;
        CVector<BaseFloat> cv1(r), cv2(c);
        cv1.SetRandn(), cv2.SetRandn();
        std::cout << "cv1:\n" << cv1;
        std::cout << "cv2:\n" << cv2;
        cm.AddVecVec(1, 0.5, cv1, cv2, kNoConj);
        std::cout << "cm = cm + (1+0.5i) * cv1 * cv2^T:\n" << cm;
        cm.AddVecVec(0.5, 2, cv1, cv2, kConj);
        std::cout << "cm = cm + (0.5+2i) * cv1 * cv2^H:\n" << cm;
    }    
}

void test_cmatrix_addmat() {
    for (int32 i = 0; i < 10; i++) {
        int32 r = Rand() % 5 + 2, c = Rand() % 5 + 2;
        CMatrix<BaseFloat> cm(r, c), m1(r, c), m2(c, r);
        cm.SetRandn(), m1.SetRandn(), m2.SetRandn();
        std::cout << "m1:\n" << m1;
        std::cout << "m2:\n" << m2;
        std::cout << "cm:\n" << cm;
        cm.AddMat(1, 1, m1, kNoTrans);
        std::cout << "cm + m1:\n" << cm;
        cm.AddMat(1, 1, m2, kTrans);
        std::cout << "cm + m2.T:\n" << cm;
    }
}

void test_cmatrix_mulelements() {
    for (int32 i = 0; i < 10; i++) {
        CMatrix<BaseFloat> cm1(4, 3), cm2(4, 3);
        cm1.SetRandn(), cm2.SetRandn();
        std::cout << cm1;
        std::cout << cm2;
        cm1.MulElements(cm2);
        std::cout << cm1;
        cm1.DivElements(cm2);
        std::cout << cm1;
        cm1.Scale(2, -1);
        std::cout << cm1;
    }
}

void test_cmatrix_invert() {
    CMatrix<BaseFloat> ans(5, 5);
    for (int32 i = 0; i < 10; i++) {
        CMatrix<BaseFloat> m(5, 5);
        m.SetRandn();
        std::cout << m;
        CMatrix<BaseFloat> m_inv(m);
        m_inv.Invert();
        ans.AddMatMat(1, 1, m, kNoTrans, m_inv, kNoTrans, 0, 0);
        std::cout << ans;
    }
}

void create_hermite_cmatrix(CMatrix<BaseFloat> *cm, MatrixIndexT s) {
    cm->Resize(s, s);
    cm->SetRandn();
    for (MatrixIndexT i = 0; i < s; i++) {
        for (MatrixIndexT j = 0; j < i; j++) {
            (*cm)(j, i, kReal) = (*cm)(i, j, kReal);
            (*cm)(j, i, kImag) = -(*cm)(i, j, kImag);
        }
        (*cm)(i, i, kImag) = 0;
    }
}


void create_hposdef_cmatrix(CMatrix<BaseFloat> *cm, int32 dim) {
    CMatrix<BaseFloat> a(dim, dim);
    cm->Resize(dim, dim);
    a.SetRandn();
    CMatrix<BaseFloat> ah(a, kTrans, kConj);
    cm->AddMatMat(1, 0, ah, kNoTrans, a, kNoTrans, 0, 0);
}

void test_cmatrix_hposdef() {
    for (int32 i = 0; i < 10; i++) {
        CMatrix<BaseFloat> a, b;
        int32 dim = Rand() % 6 + 2;
        create_hposdef_cmatrix(&a, dim);
        create_hposdef_cmatrix(&b, dim);
        CMatrix<BaseFloat> vec(dim, dim);
        Vector<BaseFloat> value(dim);
        std::cout << "A: \n" << a;
        std::cout << "B: \n" << b;
        std::cout << a.IsHermitianPosDef() << std::endl;
        std::cout << b.IsHermitianPosDef() << std::endl;
        a.Hged(&b, &value, &vec);
        std::cout << value;
        std::cout << vec;
    }       
    
}

void test_cmatrix_hed() {
    for (int32 i = 0; i < 10; i++) {
        CMatrix<BaseFloat> cm;
        int32 dim = Rand() % 6 + 2;
        create_hermite_cmatrix(&cm, dim);
        std::cout << cm;
        CMatrix<BaseFloat> eig_vectors(dim, dim), L(dim, dim), R(dim, dim);
        Vector<BaseFloat> eig_values(dim);
        cm.Hed(&eig_values, &eig_vectors);
        std::cout << eig_values;
        std::cout << eig_vectors;
        eig_vectors.Hermite();
        std::cout << eig_vectors;
        CMatrix<BaseFloat> diag(eig_values);
        L.AddMatMat(1, 1, cm, kNoTrans, eig_vectors, kNoTrans, 0, 0);
        R.AddMatMat(1, 1, eig_vectors, kNoTrans, diag, kNoTrans, 0, 0);
        L.AddMat(-1, 0, R);
        std::cout << L;
    }
}

void test_cmatrix_hermite() {
    for (int32 i = 0; i < 10; i++) {
        int32 r = Rand() % 6 + 2, c = Rand() % 6 + 2;
        CMatrix<BaseFloat> m(r, c);
        m.SetRandn();
        CMatrix<BaseFloat> trans(m), conj(m), hermite(m);
        std::cout << "Origin: \n" << m;
        trans.Transpose();
        std::cout << "Transpose: \n" << trans;
        conj.Conjugate();
        std::cout << "Conjugate: \n" << conj;
        hermite.Hermite();
        std::cout << "Hermite: \n" << hermite;
    }
}

void test_copyfromfft() {
    for (int32 i = 0; i < 10; i++) {
        int32 dim = Rand() % 8 + 2;
        Vector<BaseFloat> rv(dim * 2);
        CVector<BaseFloat> cv(dim + 1);
        rv.SetRandn();
        cv.CopyFromRealfft(rv);
        std::cout << rv;
        std::cout << cv;
    }
    for (int32 i = 0; i < 10; i++) {
        int32 r = Rand() % 8 + 2, c = Rand() % 8 + 2;
        Matrix<BaseFloat> rm(r, c * 2);
        CMatrix<BaseFloat> cm(r, c + 1);
        rm.SetRandn();
        cm.CopyFromRealfft(rm);
        std::cout << rm;
        std::cout << cm;
    }
}

int main() {
    // test_cvector_init();
    // test_cvector_addvec();
    // test_cvector_scale();
    // test_cvector_mulelements();
    // test_cvector_vecvec();
    // test_cvector_copy();
    // test_subcvector();
    // test_cmatrix_basic_op();
    // test_cmatrix_addmatmat();
    // test_cmatrix_addvecvec();
    // test_cmatrix_addmat();
    // test_cvector_addmatvec();
    // test_cmatrix_mulelements();
    // test_cmatrix_invert();
    // test_cmatrix_hed();
    // test_cmatrix_hermite();
    // test_copyfromfft();
    // test_cmatrix_scale();
    test_cmatrix_hposdef();
    return 0;
}
